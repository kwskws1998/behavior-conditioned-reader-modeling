#!/usr/bin/env python
"""Summarize residual Part 1 prediction quality across residual multiseed runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/residual_multiseed"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_root / "residual_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_runs(args.run_root, split=args.split)
    if summary.empty:
        raise FileNotFoundError(f"No residual prediction files found under {args.run_root}")
    comparisons = build_comparisons(summary)

    summary_path = output_dir / "residual_part1_summary.csv"
    comparisons_path = output_dir / "residual_part1_comparisons.csv"
    report_path = output_dir / "residual_part1_report.json"
    summary.to_csv(summary_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)
    report = {
        "summary_path": str(summary_path),
        "comparisons_path": str(comparisons_path),
        "split": args.split,
        "n_runs": int(summary[["conditioning_type", "seed"]].drop_duplicates().shape[0]),
        "n_rows": int(len(summary)),
        "comparisons": comparisons.to_dict(orient="records"),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def summarize_runs(run_root: Path, split: str) -> pd.DataFrame:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    rows = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        conditioning_type, seed = parsed
        predictions_dir = run_dir / "predictions"
        if not predictions_dir.exists():
            continue
        for profile_mode in ["actual", "mean", "shuffled"]:
            path = predictions_dir / f"{split}_{profile_mode}_predictions.csv"
            if not path.exists():
                continue
            data = pd.read_csv(path)
            if "residual_abs_error" not in data.columns:
                continue
            reconstructed_col = "reconstructed_abs_error" if "reconstructed_abs_error" in data.columns else "abs_error"
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "conditioning_type": conditioning_type,
                    "seed": int(seed),
                    "split": split,
                    "profile_mode": profile_mode,
                    "n": int(len(data)),
                    "residual_mae": float(data["residual_abs_error"].mean()),
                    "reconstructed_raw_mae": float(data[reconstructed_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def parse_run_name(name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_residual_seed(\d+)", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def build_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (conditioning_type, seed), group in summary.groupby(["conditioning_type", "seed"], sort=True):
        values = group.set_index("profile_mode")
        if {"actual", "mean"}.issubset(values.index):
            rows.append(
                comparison_row(
                    conditioning_type,
                    seed,
                    "actual_gain_vs_mean",
                    values.loc["mean"],
                    values.loc["actual"],
                )
            )
        if {"actual", "shuffled"}.issubset(values.index):
            rows.append(
                comparison_row(
                    conditioning_type,
                    seed,
                    "actual_gain_vs_shuffled",
                    values.loc["shuffled"],
                    values.loc["actual"],
                )
            )

    actual = summary.loc[summary["profile_mode"] == "actual"].copy()
    for seed, group in actual.groupby("seed", sort=True):
        values = group.set_index("conditioning_type")
        if {"concat", "moe"}.issubset(values.index):
            rows.append(
                comparison_row(
                    "moe",
                    seed,
                    "moe_gain_vs_concat_actual",
                    values.loc["concat"],
                    values.loc["moe"],
                )
            )
    return pd.DataFrame(rows)


def comparison_row(conditioning_type: str, seed: int, comparison: str, baseline: pd.Series, candidate: pd.Series) -> dict[str, object]:
    return {
        "conditioning_type": conditioning_type,
        "seed": int(seed),
        "comparison": comparison,
        "residual_mae_gain": float(baseline["residual_mae"] - candidate["residual_mae"]),
        "reconstructed_raw_mae_gain": float(baseline["reconstructed_raw_mae"] - candidate["reconstructed_raw_mae"]),
        "baseline_residual_mae": float(baseline["residual_mae"]),
        "candidate_residual_mae": float(candidate["residual_mae"]),
        "baseline_reconstructed_raw_mae": float(baseline["reconstructed_raw_mae"]),
        "candidate_reconstructed_raw_mae": float(candidate["reconstructed_raw_mae"]),
    }


if __name__ == "__main__":
    main()
