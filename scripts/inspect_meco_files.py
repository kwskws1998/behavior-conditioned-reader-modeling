#!/usr/bin/env python
"""Search MECO data files for comprehension/question/correctness columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyreadr


KEYWORDS = [
    "correct",
    "accuracy",
    "accurate",
    "acc",
    "compr",
    "question",
    "answer",
    "response",
    "score",
    "item",
    "trial",
    "sub",
    "uniform",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/meco_file_schema_report.json"))
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_files(args.data_root)
    if args.max_files > 0:
        files = files[: args.max_files]
    reports = []
    for path in files:
        try:
            reports.extend(inspect_file(path))
        except Exception as exc:
            reports.append({"path": str(path), "error": repr(exc)})

    candidate_reports = [
        report
        for report in reports
        if report.get("candidate_columns") or report.get("low_cardinality_candidate_columns")
    ]
    output = {
        "data_root": str(args.data_root),
        "n_files_scanned": len(files),
        "n_tables_scanned": len(reports),
        "n_candidate_tables": len(candidate_reports),
        "candidate_tables": candidate_reports,
        "all_tables": reports,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def discover_files(root: Path) -> list[Path]:
    suffixes = {".rda", ".rdata", ".csv", ".tsv", ".txt"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def inspect_file(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix in {".rda", ".rdata"}:
        result = pyreadr.read_r(str(path))
        reports = []
        for name, table in result.items():
            reports.append(inspect_table(path, table, table_name=name or "<unnamed>"))
        return reports
    if suffix == ".csv":
        table = pd.read_csv(path, nrows=5000)
        return [inspect_table(path, table)]
    if suffix == ".tsv":
        table = pd.read_csv(path, sep="\t", nrows=5000)
        return [inspect_table(path, table)]
    table = read_delimited_text(path)
    return [inspect_table(path, table)]


def read_delimited_text(path: Path) -> pd.DataFrame:
    for sep in ["\t", ",", ";"]:
        try:
            table = pd.read_csv(path, sep=sep, nrows=5000)
            if len(table.columns) > 1:
                return table
        except Exception:
            continue
    return pd.read_csv(path, nrows=5000)


def inspect_table(path: Path, table: pd.DataFrame, table_name: str = "") -> dict[str, object]:
    columns = [str(col) for col in table.columns]
    candidate_columns = [col for col in columns if any(keyword in col.lower() for keyword in KEYWORDS)]
    low_cardinality = []
    for col in columns:
        if not any(keyword in col.lower() for keyword in KEYWORDS):
            continue
        series = table[col].dropna()
        if series.empty:
            continue
        unique = series.unique()
        if len(unique) <= 20:
            low_cardinality.append(
                {
                    "column": col,
                    "n_unique": int(len(unique)),
                    "values": {str(key): int(value) for key, value in series.value_counts().head(20).items()},
                }
            )
    return {
        "path": str(path),
        "table_name": table_name,
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "columns": columns,
        "candidate_columns": candidate_columns,
        "low_cardinality_candidate_columns": low_cardinality,
    }


if __name__ == "__main__":
    main()
