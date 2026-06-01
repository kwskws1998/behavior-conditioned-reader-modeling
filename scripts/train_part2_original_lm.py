#!/usr/bin/env python
"""Train original Part 2 text-plus-predicted-gaze comprehension-risk models."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = PROJECT_ROOT / "artifacts" / "hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.pop("TRANSFORMERS_CACHE", None)
sys.path.insert(0, str(PROJECT_ROOT))

from part2_comprehension_risk.collator import DataCollatorForTextAuxiliaryClassification
from part2_comprehension_risk.modeling import XLMRobertaForTextAuxiliaryClassification


VARIANT_ALIASES = {
    "all": [
        "text_only",
        "text_plus_profile",
        "text_plus_mean_gaze",
        "text_plus_personalized_gaze",
        "text_plus_shuffled_gaze",
        "text_plus_profile_personalized_gaze",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--target-column", type=str, default="correct")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--aux-hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset_path)
    if "text" not in data.columns:
        raise ValueError("Part 2 dataset must contain a text column. Rebuild it with build_part2_comprehension_dataset.py.")
    output_dir = args.output_dir or args.dataset_path.parent / "original_lm_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = expand_variants(args.variants)

    all_metrics = []
    all_predictions = []
    for variant in variants:
        feature_cols = select_feature_columns(data, variant)
        run_dir = output_dir / variant
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics, predictions = train_variant(args, data, variant, feature_cols, run_dir)
        all_metrics.extend(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df.to_csv(output_dir / "part2_original_lm_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "part2_original_lm_predictions.csv", index=False)
    report = summarize_report(metrics_df)
    (output_dir / "part2_original_lm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def train_variant(
    args: argparse.Namespace,
    data: pd.DataFrame,
    variant: str,
    feature_cols: list[str],
    run_dir: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    train = data[data["split"] == "train"].copy()
    dev = data[data["split"] == "dev"].copy()
    test = data[data["split"] == "test"].copy()
    if train.empty or dev.empty:
        raise ValueError("Part 2 LM training requires non-empty train and dev splits.")
    if train[args.target_column].nunique() < 2:
        raise ValueError("Train split has only one class; cannot train BCE classifier.")

    train, dev, test, feature_stats = standardize_features(train, dev, test, feature_cols)
    local_files_only = not args.allow_downloads
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True, local_files_only=local_files_only)
    train_dataset = make_dataset(train, tokenizer, args.target_column, feature_cols, args.max_length)
    dev_dataset = make_dataset(dev, tokenizer, args.target_column, feature_cols, args.max_length)
    test_dataset = make_dataset(test, tokenizer, args.target_column, feature_cols, args.max_length) if not test.empty else None

    config = AutoConfig.from_pretrained(args.model_name, local_files_only=local_files_only)
    config.update({"aux_feature_dim": len(feature_cols), "aux_hidden_size": args.aux_hidden_size})
    model = XLMRobertaForTextAuxiliaryClassification.from_pretrained(
        args.model_name,
        config=config,
        local_files_only=local_files_only,
    )
    collator = DataCollatorForTextAuxiliaryClassification(tokenizer)
    training_args = make_training_args(args, run_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=compute_classification_metrics,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))

    metrics = []
    predictions = []
    for split_name, split_data, split_dataset in [("train", train, train_dataset), ("dev", dev, dev_dataset), ("test", test, test_dataset)]:
        if split_dataset is None or split_data.empty:
            continue
        output = trainer.predict(split_dataset, metric_key_prefix=split_name)
        metrics.append(format_metrics(variant, split_name, output.metrics, len(feature_cols), len(split_data)))
        predictions.append(make_prediction_frame(variant, split_name, split_data, output.predictions, args.target_column))

    (run_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (run_dir / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")
    return metrics, pd.concat(predictions, ignore_index=True)


def make_training_args(args: argparse.Namespace, run_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
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
    arg_names = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in arg_names else "evaluation_strategy"] = "epoch"
    if args.save_checkpoints:
        kwargs["save_strategy"] = "epoch"
        kwargs["metric_for_best_model"] = "eval_auroc"
        kwargs["greater_is_better"] = True
        kwargs["load_best_model_at_end"] = True
    else:
        kwargs["save_strategy"] = "no"
        kwargs["load_best_model_at_end"] = False
    if "use_cpu" in arg_names:
        kwargs["use_cpu"] = args.use_cpu
    else:
        kwargs["no_cuda"] = args.use_cpu
    if "save_only_model" in arg_names:
        kwargs["save_only_model"] = True
    return TrainingArguments(**kwargs)


def make_dataset(
    data: pd.DataFrame,
    tokenizer,
    target_column: str,
    feature_cols: list[str],
    max_length: int,
) -> Dataset:
    rows = []
    for _, row in data.iterrows():
        encoded = tokenizer(str(row["text"]), truncation=True, max_length=max_length)
        encoded["labels"] = float(row[target_column])
        encoded["aux_features"] = [float(row[col]) for col in feature_cols]
        rows.append(encoded)
    return Dataset.from_list(rows)


def standardize_features(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    if not feature_cols:
        return train, dev, test, {}
    train = train.copy()
    dev = dev.copy()
    test = test.copy()
    stats = {}
    for col in feature_cols:
        train_values = pd.to_numeric(train[col], errors="coerce")
        mean = float(train_values.mean()) if train_values.notna().any() else 0.0
        std = float(train_values.std()) if train_values.notna().any() else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
        for frame in [train, dev, test]:
            values = pd.to_numeric(frame[col], errors="coerce").fillna(mean)
            frame[col] = (values - mean) / std
    return train, dev, test, stats


def compute_classification_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    probabilities = sigmoid(logits)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }
    if len(np.unique(labels)) > 1:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
        metrics["log_loss"] = float(log_loss(labels, clipped))
    else:
        metrics["auroc"] = 0.5
        metrics["average_precision"] = float(labels.mean())
        metrics["log_loss"] = float("nan")
    return metrics


def format_metrics(
    variant: str,
    split: str,
    metrics: dict[str, float],
    n_features: int,
    n_rows: int,
) -> dict[str, object]:
    row = {"variant": variant, "split": split, "n": int(n_rows), "n_aux_features": int(n_features)}
    for key, value in metrics.items():
        short_key = key.removeprefix(f"{split}_")
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
            row[short_key] = float(value)
    return row


def make_prediction_frame(
    variant: str,
    split: str,
    data: pd.DataFrame,
    logits: np.ndarray,
    target_column: str,
) -> pd.DataFrame:
    probabilities = sigmoid(np.asarray(logits).reshape(-1))
    keep_cols = [col for col in ["reader_id", "trial_id", "question_id", "split", "text", target_column] if col in data.columns]
    frame = data[keep_cols].copy()
    frame["variant"] = variant
    frame["eval_split"] = split
    frame["prob_correct"] = probabilities
    frame["pred_correct"] = (probabilities >= 0.5).astype(int)
    return frame


def select_feature_columns(data: pd.DataFrame, variant: str) -> list[str]:
    profile = sorted(col for col in data.columns if col.startswith("profile_"))
    actual = sorted(col for col in data.columns if col.startswith("gaze_actual_") and not col.endswith("_observed_only"))
    mean = sorted(col for col in data.columns if col.startswith("gaze_mean_") and not col.endswith("_observed_only"))
    shuffled = sorted(col for col in data.columns if col.startswith("gaze_shuffled_") and not col.endswith("_observed_only"))
    mapping = {
        "text_only": [],
        "text_plus_profile": profile,
        "text_plus_mean_gaze": mean,
        "text_plus_personalized_gaze": actual,
        "text_plus_shuffled_gaze": shuffled,
        "text_plus_profile_personalized_gaze": profile + actual,
    }
    if variant not in mapping:
        raise ValueError(f"Unknown variant: {variant}")
    feature_cols = mapping[variant]
    if variant != "text_only" and not feature_cols:
        raise ValueError(f"No feature columns found for {variant}")
    return feature_cols


def expand_variants(values: list[str]) -> list[str]:
    expanded = []
    for value in values:
        expanded.extend(VARIANT_ALIASES.get(value, [value]))
    return list(dict.fromkeys(expanded))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def summarize_report(metrics: pd.DataFrame) -> dict[str, object]:
    test = metrics[metrics["split"] == "test"].copy()
    report = {
        "metrics_path": "part2_original_lm_metrics.csv",
        "predictions_path": "part2_original_lm_predictions.csv",
        "test_metrics": {},
        "main_comparisons": {},
    }
    for _, row in test.iterrows():
        report["test_metrics"][row["variant"]] = {
            key: float(row[key])
            for key in ["n", "accuracy", "balanced_accuracy", "auroc", "average_precision", "brier", "log_loss"]
            if key in row and pd.notna(row[key])
        }
    values = {row["variant"]: float(row["balanced_accuracy"]) for _, row in test.iterrows() if "balanced_accuracy" in row and pd.notna(row["balanced_accuracy"])}
    pairs = [
        ("text_plus_personalized_gaze", "text_only"),
        ("text_plus_personalized_gaze", "text_plus_profile"),
        ("text_plus_personalized_gaze", "text_plus_mean_gaze"),
        ("text_plus_personalized_gaze", "text_plus_shuffled_gaze"),
        ("text_plus_profile_personalized_gaze", "text_plus_personalized_gaze"),
    ]
    for left, right in pairs:
        if left in values and right in values:
            report["main_comparisons"][f"{left}_minus_{right}_balanced_accuracy"] = values[left] - values[right]
    return report


if __name__ == "__main__":
    main()
