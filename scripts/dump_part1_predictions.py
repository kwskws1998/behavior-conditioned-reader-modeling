#!/usr/bin/env python
"""Dump word-level predictions from a trained Part 1 run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.collator import DataCollatorForBehaviorConditionedTokenRegression
from part1_behavior_conditioned.data import (
    attach_residual_baseline,
    build_part1_raw_datasets,
    compute_behavior_profiles,
    ensure_log_trt_column,
    fit_profile_stats,
    load_residual_baseline_predictions,
    load_meco_rda,
    make_sentence_examples,
    normalize_profiles,
    parse_int_list,
    tokenize_and_align_examples,
)
from part1_behavior_conditioned.modeling import XLMRobertaForBehaviorConditionedTRT


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.rda_path = resolve_rda_path(args.rda_path)
    summary = json.loads((args.run_dir / "data_summary.json").read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(summary.get("seed", infer_seed(args.run_dir)))
    output_dir = args.output_dir or args.run_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_meco_rda(args.rda_path, lang=args.lang)
    data, label_column = prepare_target_data(data, summary)
    raw = build_part1_raw_datasets(
        data,
        profile_trials=summary.get("profile_trials", [1, 2]),
        train_trials=summary.get("train_trials", [3, 4, 5, 6, 7, 8, 9, 10]),
        dev_trials=summary.get("dev_trials", [11, 12]),
        test_reader_frac=0.2,
        seed=seed,
        explicit_test_readers=summary["test_readers"],
        label_column=label_column,
    )

    model_dir = args.run_dir / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True, local_files_only=True)
    model = XLMRobertaForBehaviorConditionedTRT.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()
    collator = DataCollatorForBehaviorConditionedTokenRegression(tokenizer=tokenizer)

    manifest: dict[str, str] = {}
    split_items = [("dev", raw.dev), ("test", raw.test)]
    if args.include_train:
        split_items.insert(0, ("train", build_train_profile_datasets(data, raw, summary, seed, label_column)))
    for split_name, datasets in split_items:
        for profile_mode, raw_dataset in datasets.items():
            tokenized = tokenize_and_align_examples(raw_dataset, tokenizer=tokenizer, max_length=args.max_length)
            predictions = predict_logits(model, tokenized, collator, device=device, batch_size=args.batch_size)
            output_path = output_dir / f"{split_name}_{profile_mode}_predictions.csv"
            dump_word_predictions(
                raw_dataset=raw_dataset,
                predictions=predictions,
                tokenizer=tokenizer,
                max_length=args.max_length,
                split_name=split_name,
                profile_mode=profile_mode,
                output_path=output_path,
                target_mode=summary.get("target_mode", "raw"),
            )
            manifest[f"{split_name}_{profile_mode}"] = str(output_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def infer_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else 13


def prepare_target_data(data: pd.DataFrame, summary: dict) -> tuple[pd.DataFrame, str]:
    target_mode = summary.get("target_mode", "raw")
    data = ensure_log_trt_column(data)
    if target_mode == "raw":
        return data, "log_trt"
    if target_mode != "residual":
        raise ValueError(f"Unknown target_mode in data_summary.json: {target_mode}")
    baseline_dir = summary.get("residual_baseline_dir")
    if not baseline_dir:
        raise ValueError("Residual run summary is missing residual_baseline_dir")
    baseline = load_residual_baseline_predictions(baseline_dir)
    return attach_residual_baseline(data, baseline), "residual_log_trt"


def predict_logits(
    model: XLMRobertaForBehaviorConditionedTRT,
    dataset,
    collator: DataCollatorForBehaviorConditionedTokenRegression,
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            attention_mask = batch["attention_mask"].detach().cpu().numpy()
            for row_idx in range(logits.shape[0]):
                length = int(attention_mask[row_idx].sum())
                predictions.append(logits[row_idx, :length])
    return predictions


def dump_word_predictions(
    raw_dataset,
    predictions: list[np.ndarray],
    tokenizer,
    max_length: int,
    split_name: str,
    profile_mode: str,
    output_path: Path,
    target_mode: str,
) -> None:
    rows = []
    for example_idx, example in enumerate(raw_dataset):
        encoded = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = encoded.word_ids()
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                previous_word_idx = word_idx
                continue
            previous_word_idx = word_idx
            true_value = example["labels"][word_idx]
            if true_value is None or not np.isfinite(true_value):
                continue
            pred_value = float(predictions[example_idx][token_idx])
            row = {
                "split": split_name,
                "profile_mode": profile_mode,
                "reader_id": example["reader_id"],
                "trial_id": int(example["trial_id"]),
                "sent_num": int(example["sent_num"]),
                "word_idx": int(word_idx),
                "word_num": int(example["word_nums"][word_idx]),
                "word": example["tokens"][word_idx],
            }
            if target_mode == "residual":
                baseline_value = example["baseline_log_trt"][word_idx]
                raw_value = example["raw_log_trt"][word_idx]
                if baseline_value is None or raw_value is None:
                    continue
                reconstructed = float(baseline_value) + pred_value
                residual_error = abs(pred_value - float(true_value))
                reconstructed_error = abs(reconstructed - float(raw_value))
                row.update(
                    {
                        "true_log_trt": float(raw_value),
                        "pred_log_trt": reconstructed,
                        "abs_error": reconstructed_error,
                        "baseline_pred_log_trt": float(baseline_value),
                        "true_residual_log_trt": float(true_value),
                        "pred_residual_log_trt": pred_value,
                        "reconstructed_pred_log_trt": reconstructed,
                        "residual_abs_error": residual_error,
                        "reconstructed_abs_error": reconstructed_error,
                    }
                )
            else:
                row.update(
                    {
                        "true_log_trt": float(true_value),
                        "pred_log_trt": pred_value,
                        "abs_error": abs(pred_value - float(true_value)),
                    }
                )
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_train_profile_datasets(data: pd.DataFrame, raw, summary: dict, seed: int, label_column: str) -> dict[str, Dataset]:
    raw_profiles = compute_behavior_profiles(data, profile_trials=summary.get("profile_trials", [1, 2]))
    profile_stats = fit_profile_stats(raw_profiles, raw.split.train_readers)
    profiles = normalize_profiles(raw_profiles, profile_stats)
    train_trials = summary.get("train_trials", [3, 4, 5, 6, 7, 8, 9, 10])
    return {
        mode: Dataset.from_list(
            make_sentence_examples(
                data,
                readers=raw.split.train_readers,
                trials=train_trials,
                profiles=profiles,
                profile_mode=mode,
                label_column=label_column,
                seed=seed,
            )
        )
        for mode in ["actual", "mean", "shuffled"]
    }


if __name__ == "__main__":
    main()
