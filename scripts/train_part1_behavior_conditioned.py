#!/usr/bin/env python
"""Train Part 1 behavior-conditioned TRT prediction on MECO."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.pop("TRANSFORMERS_CACHE", None)
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from part1_behavior_conditioned.collator import DataCollatorForBehaviorConditionedTokenRegression
from part1_behavior_conditioned.data import (
    attach_residual_baseline,
    build_part1_raw_datasets,
    ensure_log_trt_column,
    load_residual_baseline_predictions,
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
    parser.add_argument("--target-mode", choices=["raw", "residual"], default="raw")
    parser.add_argument("--residual-baseline-dir", type=Path, default=None)
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
    parser.add_argument("--save-checkpoints", action="store_true")
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
    data, label_column, target_summary = prepare_target_data(args, data)
    explicit_test_readers = [item.strip() for item in args.test_readers.split(",") if item.strip()] or None
    raw_datasets = build_part1_raw_datasets(
        data,
        profile_trials=parse_int_list(args.profile_trials),
        train_trials=parse_int_list(args.train_trials),
        dev_trials=parse_int_list(args.dev_trials),
        test_reader_frac=args.test_reader_frac,
        seed=args.seed,
        explicit_test_readers=explicit_test_readers,
        label_column=label_column,
    )
    label_summary = summarize_raw_dataset_labels(raw_datasets)
    if args.target_mode == "residual":
        validate_residual_labels(label_summary)
    summary = {
        "language": args.lang,
        "n_rows": int(len(data)),
        "target_mode": args.target_mode,
        "residual_baseline_dir": str(args.residual_baseline_dir) if args.residual_baseline_dir else None,
        "label_space": "raw_log1p_trt" if args.target_mode == "raw" else "residual_log1p_trt_minus_text_only_raw_prediction",
        "profile_trials": parse_int_list(args.profile_trials),
        "train_trials": parse_int_list(args.train_trials),
        "dev_trials": parse_int_list(args.dev_trials),
        "seed": args.seed,
        "conditioning_type": args.conditioning_type,
        "train_readers": raw_datasets.split.train_readers,
        "test_readers": raw_datasets.split.test_readers,
        "profile_features": raw_datasets.profile_stats.feature_names,
        "n_train_sentences": len(raw_datasets.train),
        "n_dev_sentences": {key: len(value) for key, value in raw_datasets.dev.items()},
        "n_test_sentences": {key: len(value) for key, value in raw_datasets.test.items()},
        "label_summary": label_summary,
        **target_summary,
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
        "save_total_limit": 1,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "report_to": [],
        "remove_unused_columns": True,
        "seed": args.seed,
    }
    training_arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_arg_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if args.save_checkpoints:
        training_kwargs["save_strategy"] = "epoch"
        training_kwargs["metric_for_best_model"] = "eval_mae"
        training_kwargs["greater_is_better"] = False
        training_kwargs["load_best_model_at_end"] = True
    else:
        training_kwargs["save_strategy"] = "no"
        training_kwargs["load_best_model_at_end"] = False
    if "use_cpu" in training_arg_names:
        training_kwargs["use_cpu"] = args.use_cpu
    else:
        training_kwargs["no_cuda"] = args.use_cpu
    if "save_only_model" in training_arg_names:
        training_kwargs["save_only_model"] = True
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


def prepare_target_data(args: argparse.Namespace, data):
    data = ensure_log_trt_column(data)
    if args.target_mode == "raw":
        return data, "log_trt", {}
    if args.residual_baseline_dir is None:
        raise ValueError("--residual-baseline-dir is required when --target-mode residual")
    baseline = load_residual_baseline_predictions(args.residual_baseline_dir)
    data = attach_residual_baseline(data, baseline)
    finite_residual = np.isfinite(np.asarray(data["residual_log_trt"], dtype=float))
    return (
        data,
        "residual_log_trt",
        {
            "n_residual_baseline_rows": int(len(baseline)),
            "n_rows_with_residual_label_available": int(finite_residual.sum()),
        },
    )


def summarize_raw_dataset_labels(raw_datasets) -> dict[str, dict[str, float]]:
    datasets = {"train_actual": raw_datasets.train}
    for split_name, split_datasets in [("dev", raw_datasets.dev), ("test", raw_datasets.test)]:
        for profile_mode, dataset in split_datasets.items():
            datasets[f"{split_name}_{profile_mode}"] = dataset
    return {name: summarize_dataset_labels(dataset) for name, dataset in datasets.items()}


def summarize_dataset_labels(dataset) -> dict[str, float]:
    values = []
    missing = 0
    for example in dataset:
        for value in example["labels"]:
            if value is None or not np.isfinite(value):
                missing += 1
            else:
                values.append(float(value))
    array = np.asarray(values, dtype=float)
    return {
        "n_observed": int(array.size),
        "n_missing": int(missing),
        "mean": float(array.mean()) if array.size else float("nan"),
        "std": float(array.std()) if array.size else float("nan"),
    }


def validate_residual_labels(label_summary: dict[str, dict[str, float]]) -> None:
    missing = {name: values["n_missing"] for name, values in label_summary.items() if values["n_missing"] > 0}
    if missing:
        raise ValueError(f"Residual labels are missing baseline predictions: {missing}")
    train_summary = label_summary.get("train_actual", {})
    if train_summary.get("n_observed", 0) == 0:
        raise ValueError("Residual training labels are empty")
    if train_summary.get("std", 0.0) < 1e-8:
        raise ValueError("Residual training labels have near-zero variance")


if __name__ == "__main__":
    main()
