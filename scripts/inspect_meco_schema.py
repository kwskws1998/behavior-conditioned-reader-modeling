#!/usr/bin/env python
"""Inspect MECO columns needed for Part 2 comprehension-risk modeling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import load_meco_rda


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/part2_schema_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rda_path = resolve_rda_path(args.rda_path)
    data = load_meco_rda(rda_path, lang=args.lang)
    report = {
        "rda_path": str(rda_path),
        "language": args.lang,
        "n_rows": int(len(data)),
        "n_readers": int(data["uniform_id"].nunique()),
        "n_trials": int(data["trialid"].nunique()),
        "columns": list(data.columns),
        "candidate_correctness_columns": find_candidate_correctness_columns(data),
        "candidate_question_columns": find_candidate_question_columns(data),
        "binary_or_low_cardinality_columns": summarize_low_cardinality_columns(data),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def find_candidate_correctness_columns(data: pd.DataFrame) -> list[str]:
    keywords = ["correct", "accuracy", "accurate", "acc", "compr", "question", "answer", "score"]
    candidates = []
    for col in data.columns:
        name = str(col).lower()
        if any(keyword in name for keyword in keywords):
            candidates.append(str(col))
    return candidates


def find_candidate_question_columns(data: pd.DataFrame) -> list[str]:
    keywords = ["question", "qid", "qnum", "q_id", "answer", "response", "item"]
    return [str(col) for col in data.columns if any(keyword in str(col).lower() for keyword in keywords)]


def summarize_low_cardinality_columns(data: pd.DataFrame, max_unique: int = 8) -> list[dict[str, object]]:
    rows = []
    for col in data.columns:
        series = data[col].dropna()
        if series.empty:
            continue
        unique = series.unique()
        if len(unique) <= max_unique:
            counts = series.value_counts(dropna=False).head(max_unique)
            rows.append(
                {
                    "column": str(col),
                    "n_unique": int(len(unique)),
                    "values": {str(key): int(value) for key, value in counts.items()},
                }
            )
    return rows


if __name__ == "__main__":
    main()
