#!/usr/bin/env python
"""Run Part 1 experiments across multiple seeds and conditioning types."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/multiseed"))
    parser.add_argument("--conditioning-types", nargs="+", default=["none", "concat", "moe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_script = Path(__file__).resolve().parent / "train_part1_behavior_conditioned.py"
    for conditioning_type in args.conditioning_types:
        for seed in args.seeds:
            output_dir = args.output_root / f"{conditioning_type}_seed{seed}"
            command = [
                sys.executable,
                str(train_script),
                "--rda-path",
                str(args.rda_path),
                "--output-dir",
                str(output_dir),
                "--conditioning-type",
                conditioning_type,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
            ]
            if conditioning_type == "moe":
                command.extend(["--num-experts", str(args.num_experts)])
            if args.use_cpu:
                command.append("--use-cpu")
            if args.allow_downloads:
                command.append("--allow-downloads")
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

