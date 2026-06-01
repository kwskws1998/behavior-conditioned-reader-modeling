#!/usr/bin/env python
"""Run prediction dumping and high-variance analysis for multiple Part 1 runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/multiseed"))
    parser.add_argument("--rda-path", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--skip-dump", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dump_script = script_dir / "dump_part1_predictions.py"
    analysis_script = script_dir / "analyze_high_variance.py"
    run_dirs = discover_run_dirs(args.run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs with best_model/ and data_summary.json found under {args.run_root}")

    summary_rows = []
    for run_dir in run_dirs:
        if not args.skip_dump:
            command = [
                sys.executable,
                str(dump_script),
                "--run-dir",
                str(run_dir),
                "--rda-path",
                str(args.rda_path),
                "--batch-size",
                str(args.batch_size),
            ]
            if args.use_cpu:
                command.append("--use-cpu")
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)

        for split in args.splits:
            command = [
                sys.executable,
                str(analysis_script),
                "--run-dir",
                str(run_dir),
                "--rda-path",
                str(args.rda_path),
                "--split",
                split,
            ]
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            report_path = run_dir / "high_variance" / f"{split}_high_variance_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            model, seed = parse_model_seed(run_dir.name)
            summary_rows.append(
                {
                    "run_dir": str(run_dir),
                    "model": model,
                    "seed": seed,
                    "split": split,
                    **report["overall"],
                    **report["high_low_contrast"],
                }
            )

    output_path = args.run_root / "high_variance_summary.csv"
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def discover_run_dirs(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and (path / "best_model").exists() and (path / "data_summary.json").exists()
    )


def parse_model_seed(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


if __name__ == "__main__":
    main()

