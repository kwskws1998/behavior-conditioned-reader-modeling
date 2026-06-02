#!/usr/bin/env python
"""Run VAD-token raw baselines and full-profile residual Part 1 models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, required=True)
    parser.add_argument("--vad-features-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/vad_residual_multiseed"))
    parser.add_argument("--conditioning-types", nargs="+", default=["concat", "moe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42, 87, 100])
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--token-feature-set", choices=["vad_word", "vad_word_sentence"], default="vad_word_sentence")
    parser.add_argument("--profile-feature-set", choices=["full_gaze", "full_gaze_vad"], default="full_gaze_vad")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_part1_behavior_conditioned.py"
    dump_script = script_dir / "dump_part1_predictions.py"
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        raw_output = args.output_root / f"none_raw_vad_seed{seed}"
        run_train(
            train_script=train_script,
            rda_path=args.rda_path,
            vad_features_path=args.vad_features_path,
            output_dir=raw_output,
            conditioning_type="none",
            target_mode="raw",
            seed=seed,
            args=args,
            profile_feature_set="behavior_only",
        )
        run_dump(dump_script=dump_script, rda_path=args.rda_path, run_dir=raw_output, args=args)

        for conditioning_type in args.conditioning_types:
            model_prefix = "moe2" if conditioning_type == "moe" and args.num_experts == 2 else conditioning_type
            residual_output = args.output_root / f"{model_prefix}_full_vad_residual_seed{seed}"
            run_train(
                train_script=train_script,
                rda_path=args.rda_path,
                vad_features_path=args.vad_features_path,
                output_dir=residual_output,
                conditioning_type=conditioning_type,
                target_mode="residual",
                seed=seed,
                args=args,
                residual_baseline_dir=raw_output / "predictions",
                profile_feature_set=args.profile_feature_set,
            )
            run_dump(dump_script=dump_script, rda_path=args.rda_path, run_dir=residual_output, args=args)


def run_train(
    train_script: Path,
    rda_path: Path,
    vad_features_path: Path,
    output_dir: Path,
    conditioning_type: str,
    target_mode: str,
    seed: int,
    args: argparse.Namespace,
    profile_feature_set: str,
    residual_baseline_dir: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(train_script),
        "--rda-path",
        str(rda_path),
        "--vad-features-path",
        str(vad_features_path),
        "--output-dir",
        str(output_dir),
        "--conditioning-type",
        conditioning_type,
        "--target-mode",
        target_mode,
        "--profile-feature-set",
        profile_feature_set,
        "--token-feature-set",
        args.token_feature_set,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
    ]
    if residual_baseline_dir is not None:
        command.extend(["--residual-baseline-dir", str(residual_baseline_dir)])
    if conditioning_type == "moe":
        command.extend(["--num-experts", str(args.num_experts)])
    if args.use_cpu:
        command.append("--use-cpu")
    if args.allow_downloads:
        command.append("--allow-downloads")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_dump(dump_script: Path, rda_path: Path, run_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(dump_script),
        "--run-dir",
        str(run_dir),
        "--rda-path",
        str(rda_path),
        "--vad-features-path",
        str(args.vad_features_path),
        "--batch-size",
        str(args.batch_size),
        "--include-train",
    ]
    if args.use_cpu:
        command.append("--use-cpu")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
