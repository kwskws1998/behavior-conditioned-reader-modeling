#!/usr/bin/env python
"""Train Part 1 behavior-conditioned TRT prediction on MECO."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.pop("TRANSFORMERS_CACHE", None)
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from part1_behavior_conditioned.collator import DataCollatorForBehaviorConditionedTokenRegression
from part1_behavior_conditioned.data import (
    build_part1_raw_datasets,
    load_meco_rda,
    parse_int_list,
    tokenize_and_align_examples,
)
from part1_behavior_conditioned.metrics import compute_token_regression_metrics
from part1_behavior_conditioned.modeling import XLMRobertaForBehaviorConditionedTRT


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rda-path",
        type=Path,
        default=PROJECT_ROOT / "version 1.3" / "primary data" / "eye tracking data" / "joint_l1_data_trimmed_version1.3.rda",
    )
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "part1_behavior_conditioned")
    parser.add_argument("--profile-trials", type=str, default="1,2")
    parser.add_argument("--train-trials", type=str, default="3,4,5,6,7,8,9,10")
    parser.add_argument("--dev-trials", type=str, default="11,12")
    parser.add_argument("--test-reader-frac", type=float, default=0.2)
    parser.add_argument("--test-readers", type=str, default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--conditioning-type", choices=["none", "concat", "moe"], default="concat")
    parser.add_argument("--profile-hidden-size", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--expert-hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit-train-sentences", type=int, default=0)
    parser.add_argument("--limit-eval-sentences", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        print(f"RDA path not found at {path}; using discovered file {candidates[0]}", flush=True)
        return candidates[0]
    if len(candidates) > 1:
        candidate_list = "\n".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"RDA path not found at {path}. Multiple candidates found:\n{candidate_list}")
    raise FileNotFoundError(
        f"RDA path not found at {path}. Run `find data -name '{TARGET_RDA_NAME}' -print` "
        "or unzip the MECO data archive under data/."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.rda_path = resolve_rda_path(args.rda_path)
    data = load_meco_rda(args.rda_path, lang=args.lang)
    explicit_test_readers = [item.strip() for item in args.test_readers.split(",") if item.strip()] or None
    raw_datasets = build_part1_raw_datasets(
        data,
        profile_trials=parse_int_list(args.profile_trials),
        train_trials=parse_int_list(args.train_trials),
        dev_trials=parse_int_list(args.dev_trials),
        test_reader_frac=args.test_reader_frac,
        seed=args.seed,
        explicit_test_readers=explicit_test_readers,
    )
    summary = {
        "language": args.lang,
        "n_rows": int(len(data)),
        "profile_trials": parse_int_list(args.profile_trials),
        "train_trials": parse_int_list(args.train_trials),
        "dev_trials": parse_int_list(args.dev_trials),
        "train_readers": raw_datasets.split.train_readers,
        "test_readers": raw_datasets.split.test_readers,
        "profile_features": raw_datasets.profile_stats.feature_names,
        "n_train_sentences": len(raw_datasets.train),
        "n_dev_sentences": {key: len(value) for key, value in raw_datasets.dev.items()},
        "n_test_sentences": {key: len(value) for key, value in raw_datasets.test.items()},
    }
    (args.output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    if args.limit_train_sentences > 0:
        raw_datasets = raw_datasets.__class__(
            train=raw_datasets.train.select(range(min(args.limit_train_sentences, len(raw_datasets.train)))),
            dev=raw_datasets.dev,
            test=raw_datasets.test,
            split=raw_datasets.split,
            profile_stats=raw_datasets.profile_stats,
        )
    if args.limit_eval_sentences > 0:
        raw_datasets = raw_datasets.__class__(
            train=raw_datasets.train,
            dev={
                key: value.select(range(min(args.limit_eval_sentences, len(value))))
                for key, value in raw_datasets.dev.items()
            },
            test={
                key: value.select(range(min(args.limit_eval_sentences, len(value))))
                for key, value in raw_datasets.test.items()
            },
            split=raw_datasets.split,
            profile_stats=raw_datasets.profile_stats,
        )

    local_files_only = not args.allow_downloads
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        add_prefix_space=True,
        local_files_only=local_files_only,
    )
    train_dataset = tokenize_and_align_examples(raw_datasets.train, tokenizer=tokenizer, max_length=args.max_length)
    dev_datasets = {
        key: tokenize_and_align_examples(value, tokenizer=tokenizer, max_length=args.max_length)
        for key, value in raw_datasets.dev.items()
    }
    test_datasets = {
        key: tokenize_and_align_examples(value, tokenizer=tokenizer, max_length=args.max_length)
        for key, value in raw_datasets.test.items()
    }

    profile_dim = len(raw_datasets.profile_stats.feature_names)
    config = AutoConfig.from_pretrained(args.model_name, local_files_only=local_files_only)
    config.update(
        {
            "profile_dim": profile_dim,
            "profile_hidden_size": args.profile_hidden_size,
            "conditioning_type": args.conditioning_type,
            "num_experts": args.num_experts,
            "expert_hidden_size": args.expert_hidden_size,
        }
    )
    model = XLMRobertaForBehaviorConditionedTRT.from_pretrained(
        args.model_name,
        config=config,
        local_files_only=local_files_only,
    )
    collator = DataCollatorForBehaviorConditionedTokenRegression(tokenizer=tokenizer)

    training_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "report_to": [],
        "metric_for_best_model": "eval_mae",
        "greater_is_better": False,
        "load_best_model_at_end": True,
        "remove_unused_columns": True,
        "seed": args.seed,
    }
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_arg_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "use_cpu" in training_arg_names:
        training_kwargs["use_cpu"] = args.use_cpu
    else:
        training_kwargs["no_cuda"] = args.use_cpu
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets["actual"],
        data_collator=collator,
        compute_metrics=compute_token_regression_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    all_metrics: dict[str, dict[str, float]] = {}
    for split_name, datasets in [("dev", dev_datasets), ("test", test_datasets)]:
        for profile_mode, dataset in datasets.items():
            metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=f"{split_name}_{profile_mode}")
            all_metrics[f"{split_name}_{profile_mode}"] = {key: float(value) for key, value in metrics.items()}
    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
