#!/usr/bin/env python
"""Analyze whether personalization gains are larger on high-variance words."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import load_meco_rda


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--stratify-by", choices=["word_variance", "reader_deviation"], default="word_variance")
    parser.add_argument("--min-readers", type=int, default=5)
    parser.add_argument("--num-bins", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.rda_path = resolve_rda_path(args.rda_path)
    predictions_dir = args.predictions_dir or args.run_dir / "predictions"
    output_dir = args.output_dir or args.run_dir / "high_variance"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run_dir / "data_summary.json").read_text(encoding="utf-8"))
    data = load_meco_rda(args.rda_path, lang=args.lang)
    item_stats = compute_train_reader_word_stats(
        data=data,
        train_readers=summary["train_readers"],
        trials=summary.get("dev_trials", [11, 12]),
        min_readers=args.min_readers,
    )

    merged = load_and_merge_predictions(predictions_dir, split=args.split)
    merged = merged.merge(item_stats, on=["trial_id", "sent_num", "word_num"], how="left")
    merged = merged.dropna(subset=["word_mean_log_trt", "word_std_log_trt", "word_var_log_trt"]).copy()
    merged["reader_deviation_z"] = (merged["true_log_trt"] - merged["word_mean_log_trt"]) / merged["word_std_log_trt"].clip(lower=1e-6)
    merged["abs_reader_deviation_z"] = merged["reader_deviation_z"].abs()
    merged["gain_vs_mean"] = merged["abs_error_mean"] - merged["abs_error_actual"]
    merged["gain_vs_shuffled"] = merged["abs_error_shuffled"] - merged["abs_error_actual"]
    merged["actual_beats_mean"] = merged["gain_vs_mean"] > 0
    merged["actual_beats_shuffled"] = merged["gain_vs_shuffled"] > 0
    stratify_column = "word_var_log_trt" if args.stratify_by == "word_variance" else "abs_reader_deviation_z"
    merged["stratify_value"] = merged[stratify_column]
    merged["stratify_bin"] = make_bins(merged["stratify_value"], args.num_bins)

    bins = summarize_bins(merged)
    contrast = summarize_high_low_contrast(bins)

    prefix = f"{args.split}_{args.stratify_by}"
    merged.to_csv(output_dir / f"{prefix}_word_level.csv", index=False)
    bins.to_csv(output_dir / f"{prefix}_bins.csv", index=False)
    report = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "stratify_by": args.stratify_by,
        "stratify_column": stratify_column,
        "n_rows": int(len(merged)),
        "min_readers": args.min_readers,
        "num_bins": args.num_bins,
        "overall": summarize_overall(merged),
        "high_low_contrast": contrast,
    }
    (output_dir / f"{prefix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def compute_train_reader_word_stats(
    data: pd.DataFrame,
    train_readers: list[str],
    trials: list[int],
    min_readers: int,
) -> pd.DataFrame:
    subset = data.loc[
        data["uniform_id"].isin(train_readers)
        & data["trialid"].isin(trials)
        & data["dur"].notna()
    ].copy()
    subset["log_trt"] = np.log1p(subset["dur"].astype(float))
    grouped = subset.groupby(["trialid", "sentnum", "wordnum"], as_index=False).agg(
        word=("word", "first"),
        n_train_readers=("uniform_id", "nunique"),
        word_mean_log_trt=("log_trt", "mean"),
        word_var_log_trt=("log_trt", "var"),
        word_std_log_trt=("log_trt", "std"),
    )
    grouped = grouped.loc[grouped["n_train_readers"] >= min_readers].copy()
    grouped = grouped.rename(columns={"trialid": "trial_id", "sentnum": "sent_num", "wordnum": "word_num"})
    grouped["word_var_log_trt"] = grouped["word_var_log_trt"].fillna(0.0)
    grouped["word_std_log_trt"] = grouped["word_std_log_trt"].fillna(0.0)
    return grouped


def load_and_merge_predictions(predictions_dir: Path, split: str) -> pd.DataFrame:
    actual = read_prediction_file(predictions_dir / f"{split}_actual_predictions.csv", "actual")
    mean = read_prediction_file(predictions_dir / f"{split}_mean_predictions.csv", "mean")
    shuffled = read_prediction_file(predictions_dir / f"{split}_shuffled_predictions.csv", "shuffled")
    keys = ["reader_id", "trial_id", "sent_num", "word_num", "word_idx", "word", "true_log_trt"]
    merged = actual.merge(mean, on=keys, how="inner").merge(shuffled, on=keys, how="inner")
    return merged


def read_prediction_file(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}. Run scripts/dump_part1_predictions.py first.")
    data = pd.read_csv(path)
    keep = ["reader_id", "trial_id", "sent_num", "word_num", "word_idx", "word", "true_log_trt", "pred_log_trt", "abs_error"]
    data = data[keep].copy()
    return data.rename(
        columns={
            "pred_log_trt": f"pred_log_trt_{suffix}",
            "abs_error": f"abs_error_{suffix}",
        }
    )


def make_bins(values: pd.Series, num_bins: int) -> pd.Series:
    try:
        return pd.qcut(values, q=num_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        ranks = values.rank(method="average")
        return pd.qcut(ranks, q=num_bins, labels=False, duplicates="drop") + 1


def summarize_bins(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bin_id, group in data.groupby("stratify_bin", sort=True):
        rows.append(
            {
                "stratify_bin": int(bin_id),
                "n": int(len(group)),
                "mean_stratify_value": float(group["stratify_value"].mean()),
                "mean_word_var_log_trt": float(group["word_var_log_trt"].mean()),
                "mean_abs_reader_deviation_z": float(group["abs_reader_deviation_z"].mean()),
                "mean_abs_error_actual": float(group["abs_error_actual"].mean()),
                "mean_abs_error_mean": float(group["abs_error_mean"].mean()),
                "mean_abs_error_shuffled": float(group["abs_error_shuffled"].mean()),
                "gain_vs_mean": float(group["gain_vs_mean"].mean()),
                "gain_vs_shuffled": float(group["gain_vs_shuffled"].mean()),
                "win_rate_vs_mean": float(group["actual_beats_mean"].mean()),
                "win_rate_vs_shuffled": float(group["actual_beats_shuffled"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_high_low_contrast(bins: pd.DataFrame) -> dict[str, float]:
    low = bins.sort_values("stratify_bin").iloc[0]
    high = bins.sort_values("stratify_bin").iloc[-1]
    return {
        "high_bin": int(high["stratify_bin"]),
        "low_bin": int(low["stratify_bin"]),
        "gain_vs_mean_high_minus_low": float(high["gain_vs_mean"] - low["gain_vs_mean"]),
        "gain_vs_shuffled_high_minus_low": float(high["gain_vs_shuffled"] - low["gain_vs_shuffled"]),
        "win_rate_vs_mean_high_minus_low": float(high["win_rate_vs_mean"] - low["win_rate_vs_mean"]),
        "win_rate_vs_shuffled_high_minus_low": float(high["win_rate_vs_shuffled"] - low["win_rate_vs_shuffled"]),
    }


def summarize_overall(data: pd.DataFrame) -> dict[str, float]:
    return {
        "mae_actual": float(data["abs_error_actual"].mean()),
        "mae_mean": float(data["abs_error_mean"].mean()),
        "mae_shuffled": float(data["abs_error_shuffled"].mean()),
        "gain_vs_mean": float(data["gain_vs_mean"].mean()),
        "gain_vs_shuffled": float(data["gain_vs_shuffled"].mean()),
        "win_rate_vs_mean": float(data["actual_beats_mean"].mean()),
        "win_rate_vs_shuffled": float(data["actual_beats_shuffled"].mean()),
    }


if __name__ == "__main__":
    main()
