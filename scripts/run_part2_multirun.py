#!/usr/bin/env python
"""Run Part 2 comprehension-risk modeling for multiple MoE Part 1 runs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/multiseed"))
    parser.add_argument("--rda-path", type=Path, required=True)
    parser.add_argument("--correctness-column", type=str, default="")
    parser.add_argument("--question-column", type=str, default="")
    parser.add_argument("--trainer", choices=["original_lm", "linear_backup"], default="original_lm")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--skip-dump", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dump_script = script_dir / "dump_part1_predictions.py"
    build_script = script_dir / "build_part2_comprehension_dataset.py"
    train_script = script_dir / (
        "train_part2_original_lm.py" if args.trainer == "original_lm" else "train_part2_comprehension_risk.py"
    )
    run_dirs = discover_moe_run_dirs(args.run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No MoE run dirs found under {args.run_root}")

    summary_frames = []
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
                "--include-train",
            ]
            if args.use_cpu:
                command.append("--use-cpu")
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)

        dataset_path = run_dir / "part2_comprehension" / "part2_dataset.csv"
        command = [
            sys.executable,
            str(build_script),
            "--part1-run-dir",
            str(run_dir),
            "--rda-path",
            str(args.rda_path),
            "--output-path",
            str(dataset_path),
        ]
        if args.correctness_column:
            command.extend(["--correctness-column", args.correctness_column])
        if args.question_column:
            command.extend(["--question-column", args.question_column])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        model_dir = run_dir / "part2_comprehension" / args.trainer
        command = [
            sys.executable,
            str(train_script),
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(model_dir),
        ]
        if args.trainer == "original_lm":
            command.extend(["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)])
            if args.allow_downloads:
                command.append("--allow-downloads")
            if args.use_cpu:
                command.append("--use-cpu")
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

        metrics_path = (
            model_dir / "part2_original_lm_metrics.csv"
            if args.trainer == "original_lm"
            else model_dir / "part2_metrics.csv"
        )
        metrics = pd.read_csv(metrics_path)
        model_name, seed = parse_model_seed(run_dir.name)
        metrics.insert(0, "run_dir", str(run_dir))
        metrics.insert(1, "part1_model", model_name)
        metrics.insert(2, "seed", seed)
        summary_frames.append(metrics)

    summary = pd.concat(summary_frames, ignore_index=True)
    output_path = args.run_root / f"part2_{args.trainer}_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def discover_moe_run_dirs(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and path.name.startswith("moe_seed")
        and (path / "best_model").exists()
        and (path / "data_summary.json").exists()
    )


def parse_model_seed(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


if __name__ == "__main__":
    main()
