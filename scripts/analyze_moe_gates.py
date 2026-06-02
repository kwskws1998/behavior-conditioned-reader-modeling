#!/usr/bin/env python
"""Analyze whether MoE gates encode stable reader-type structure."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import (
    attach_vad_features,
    compute_behavior_profiles,
    fit_profile_stats,
    load_vad_features,
    load_meco_rda,
    normalize_profiles,
)
from part1_behavior_conditioned.modeling import XLMRobertaForBehaviorConditionedTRT


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/multiseed"))
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--vad-features-path", type=Path, default=None)
    parser.add_argument("--use-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.rda_path = resolve_rda_path(args.rda_path)
    run_dirs = [args.run_dir] if args.run_dir else discover_moe_run_dirs(args.run_root)
    if not run_dirs:
        raise FileNotFoundError(f"No MoE run dirs found under {args.run_root}")

    all_gate_rows = []
    all_corr_rows = []
    all_stability_rows = []
    all_pca_rows = []
    all_expert_profile_rows = []
    all_gain_corr_rows = []
    all_expert_gain_rows = []

    data = load_meco_rda(args.rda_path, lang=args.lang)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    for run_dir in run_dirs:
        if args.output_dir and args.run_dir is None:
            output_dir = args.output_dir / run_dir.name
        else:
            output_dir = args.output_dir or run_dir / "gate_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = analyze_run(
            run_dir=run_dir,
            data=data,
            output_dir=output_dir,
            device=device,
            vad_features_path=args.vad_features_path,
        )
        all_gate_rows.append(result["gates"])
        all_corr_rows.append(result["correlations"])
        all_stability_rows.append(result["stability"])
        all_pca_rows.append(result["pca"])
        all_expert_profile_rows.append(result["expert_profiles"])
        if result["gain_correlations"] is not None:
            all_gain_corr_rows.append(result["gain_correlations"])
        if result["expert_gains"] is not None:
            all_expert_gain_rows.append(result["expert_gains"])

    if args.run_dir is None:
        aggregate_dir = args.run_root / "gate_analysis"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(all_gate_rows, ignore_index=True).to_csv(aggregate_dir / "all_moe_gates.csv", index=False)
        pd.concat(all_corr_rows, ignore_index=True).to_csv(aggregate_dir / "all_gate_behavior_correlations.csv", index=False)
        pd.concat(all_stability_rows, ignore_index=True).to_csv(aggregate_dir / "all_gate_stability.csv", index=False)
        pd.concat(all_pca_rows, ignore_index=True).to_csv(aggregate_dir / "all_gate_pca.csv", index=False)
        pd.concat(all_expert_profile_rows, ignore_index=True).to_csv(
            aggregate_dir / "all_dominant_expert_behavior_profiles.csv",
            index=False,
        )
        if all_gain_corr_rows:
            pd.concat(all_gain_corr_rows, ignore_index=True).to_csv(aggregate_dir / "all_gate_gain_correlations.csv", index=False)
        if all_expert_gain_rows:
            pd.concat(all_expert_gain_rows, ignore_index=True).to_csv(aggregate_dir / "all_dominant_expert_gains.csv", index=False)
        aggregate_report = summarize_across_runs(
            pd.concat(all_stability_rows, ignore_index=True),
            pd.concat(all_corr_rows, ignore_index=True),
            all_gain_corr_rows,
        )
        (aggregate_dir / "gate_analysis_report.json").write_text(json.dumps(aggregate_report, indent=2), encoding="utf-8")
        print(json.dumps(aggregate_report, indent=2))


def analyze_run(
    run_dir: Path,
    data: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
    vad_features_path: Path | None,
) -> dict[str, pd.DataFrame | None]:
    summary = json.loads((run_dir / "data_summary.json").read_text(encoding="utf-8"))
    run_data = prepare_vad_data(data, summary, vad_features_path)
    model_dir = run_dir / "best_model"
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if getattr(config, "conditioning_type", None) != "moe":
        raise ValueError(f"{run_dir} is not a MoE run")
    model = XLMRobertaForBehaviorConditionedTRT.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()

    train_readers = summary["train_readers"]
    test_readers = summary["test_readers"]
    all_readers = sorted(train_readers + test_readers)
    profile_trials = summary.get("profile_trials", [1, 2])
    raw_profiles = compute_behavior_profiles(
        run_data,
        profile_trials=profile_trials,
        profile_feature_set=summary.get("profile_feature_set", "behavior_only"),
    )
    profile_stats = fit_profile_stats(raw_profiles, train_readers)
    normalized_profiles = normalize_profiles(raw_profiles, profile_stats)
    gates = compute_gate_dataframe(
        model=model,
        raw_profiles=raw_profiles,
        normalized_profiles=normalized_profiles,
        readers=all_readers,
        train_readers=set(train_readers),
        device=device,
        run_dir=run_dir,
        profile_label="actual",
    )
    correlations = compute_gate_behavior_correlations(gates, summary)
    pca = compute_gate_pca(gates, run_dir)
    expert_profiles = summarize_dominant_expert_behavior_profiles(gates, run_dir)
    stability = compute_gate_stability(
        model=model,
        data=run_data,
        profile_stats=profile_stats,
        profile_feature_set=summary.get("profile_feature_set", "behavior_only"),
        readers=all_readers,
        train_readers=set(train_readers),
        device=device,
        run_dir=run_dir,
    )
    reader_gains = load_reader_gains(run_dir)
    gain_correlations = None
    expert_gains = None
    if reader_gains is not None:
        gate_gains = gates.merge(reader_gains, on="reader_id", how="inner")
        gate_gains.to_csv(output_dir / "reader_gate_gains.csv", index=False)
        gain_correlations = compute_gate_gain_correlations(gate_gains, run_dir)
        expert_gains = summarize_dominant_expert_gains(gate_gains, run_dir)

    gates.to_csv(output_dir / "moe_gates.csv", index=False)
    correlations.to_csv(output_dir / "gate_behavior_correlations.csv", index=False)
    pca.to_csv(output_dir / "gate_pca.csv", index=False)
    expert_profiles.to_csv(output_dir / "dominant_expert_behavior_profiles.csv", index=False)
    stability.to_csv(output_dir / "gate_stability.csv", index=False)
    if gain_correlations is not None:
        gain_correlations.to_csv(output_dir / "gate_gain_correlations.csv", index=False)
    if expert_gains is not None:
        expert_gains.to_csv(output_dir / "dominant_expert_gains.csv", index=False)

    report = {
        "run_dir": str(run_dir),
        "seed": parse_model_seed(run_dir.name)[1],
        "num_readers": int(len(gates)),
        "num_experts": int(getattr(config, "num_experts", 0)),
        "mean_normalized_gate_entropy": float(gates["gate_entropy_norm"].mean()),
        "mean_same_reader_trial1_trial2_cosine": float(stability["same_reader_cosine_t1_t2"].mean()),
        "mean_different_reader_cosine": float(stability["different_reader_cosine_mean"].mean()),
        "mean_same_minus_different_cosine": float(stability["same_minus_different_cosine"].mean()),
        "mean_same_reader_trial1_trial2_l2": float(stability["same_reader_l2_t1_t2"].mean()),
        "mean_different_reader_l2": float(stability["different_reader_l2_mean"].mean()),
        "mean_different_minus_same_l2": float(stability["different_minus_same_l2"].mean()),
        "top_abs_gate_behavior_correlations": top_abs_correlations(correlations, top_k=12),
    }
    (output_dir / "gate_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return {
        "gates": gates,
        "correlations": correlations,
        "pca": pca,
        "expert_profiles": expert_profiles,
        "stability": stability,
        "gain_correlations": gain_correlations,
        "expert_gains": expert_gains,
    }


def prepare_vad_data(data: pd.DataFrame, summary: dict, override_path: Path | None) -> pd.DataFrame:
    token_feature_set = summary.get("token_feature_set", "none")
    profile_feature_set = summary.get("profile_feature_set", "behavior_only")
    needs_vad = token_feature_set != "none" or profile_feature_set == "full_gaze_vad"
    if not needs_vad:
        return data
    vad_path = override_path or summary.get("vad_features_path")
    if not vad_path:
        raise ValueError(f"{summary.get('conditioning_type', 'run')} requires VAD features but no path was provided")
    return attach_vad_features(data, load_vad_features(vad_path))


def compute_gate_dataframe(
    model: XLMRobertaForBehaviorConditionedTRT,
    raw_profiles: pd.DataFrame,
    normalized_profiles: dict[str, np.ndarray],
    readers: list[str],
    train_readers: set[str],
    device: torch.device,
    run_dir: Path,
    profile_label: str,
) -> pd.DataFrame:
    gate_matrix = compute_gates(model, [normalized_profiles[reader] for reader in readers], device=device)
    rows = []
    model_name, seed = parse_model_seed(run_dir.name)
    for idx, reader in enumerate(readers):
        gate = gate_matrix[idx]
        row = {
            "run_dir": str(run_dir),
            "model": model_name,
            "seed": seed,
            "reader_id": reader,
            "reader_split": "train" if reader in train_readers else "test",
            "profile_label": profile_label,
            "gate_entropy": entropy(gate),
            "gate_entropy_norm": entropy(gate) / math.log(len(gate)),
            "dominant_expert": int(np.argmax(gate)),
            "dominant_weight": float(np.max(gate)),
        }
        for feature_name, value in raw_profiles.loc[reader].items():
            row[f"behavior_{feature_name}"] = float(value)
        for expert_idx, value in enumerate(gate):
            row[f"gate_{expert_idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_gates(
    model: XLMRobertaForBehaviorConditionedTRT,
    profiles: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    if not hasattr(model, "gate"):
        raise ValueError("Model does not have a MoE gate")
    profile_tensor = torch.tensor(np.stack(profiles), dtype=torch.float32, device=device)
    with torch.no_grad():
        profile_repr = model.profile_encoder(profile_tensor)
        gates = torch.softmax(model.gate(profile_repr), dim=-1)
    return gates.detach().cpu().numpy()


def compute_gate_behavior_correlations(gates: pd.DataFrame, summary: dict) -> pd.DataFrame:
    behavior_cols = [col for col in gates.columns if col.startswith("behavior_")]
    gate_cols = gate_weight_cols(gates)
    rows = []
    for subset_name, subset in [
        ("all", gates),
        ("train", gates[gates["reader_id"].isin(summary["train_readers"])]),
        ("test", gates[gates["reader_id"].isin(summary["test_readers"])]),
    ]:
        if subset.empty:
            continue
        run_dir = subset["run_dir"].iloc[0]
        model_name = subset["model"].iloc[0]
        seed = subset["seed"].iloc[0]
        for gate_col in gate_cols:
            for behavior_col in behavior_cols:
                rows.append(
                    {
                        "run_dir": run_dir,
                        "model": model_name,
                        "seed": seed,
                        "subset": subset_name,
                        "gate": gate_col,
                        "behavior_feature": behavior_col.removeprefix("behavior_"),
                        "n": int(len(subset)),
                        "pearson": safe_corr(subset[gate_col], subset[behavior_col]),
                        "spearman": safe_corr(subset[gate_col].rank(), subset[behavior_col].rank()),
                    }
                )
    return pd.DataFrame(rows)


def compute_gate_pca(gates: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    gate_cols = gate_weight_cols(gates)
    matrix = gates[gate_cols].to_numpy(dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    total_variance = float((singular_values**2).sum())
    explained = (singular_values[:2] ** 2) / total_variance if total_variance > 1e-12 else np.zeros(2)

    model_name, seed = parse_model_seed(run_dir.name)
    rows = []
    for idx, source_row in gates.reset_index(drop=True).iterrows():
        row = {
            "run_dir": str(run_dir),
            "model": model_name,
            "seed": seed,
            "reader_id": source_row["reader_id"],
            "reader_split": source_row["reader_split"],
            "dominant_expert": int(source_row["dominant_expert"]),
            "dominant_weight": float(source_row["dominant_weight"]),
            "gate_entropy_norm": float(source_row["gate_entropy_norm"]),
            "pc1": float(coords[idx, 0]) if coords.shape[1] > 0 else 0.0,
            "pc2": float(coords[idx, 1]) if coords.shape[1] > 1 else 0.0,
            "pc1_explained_variance": float(explained[0]) if len(explained) > 0 else 0.0,
            "pc2_explained_variance": float(explained[1]) if len(explained) > 1 else 0.0,
        }
        for col in [col for col in gates.columns if col.startswith("behavior_")]:
            row[col] = float(source_row[col])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_dominant_expert_behavior_profiles(gates: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    behavior_cols = [col for col in gates.columns if col.startswith("behavior_")]
    gate_cols = gate_weight_cols(gates)
    model_name, seed = parse_model_seed(run_dir.name)
    rows = []
    for split_name, subset in [("all", gates), ("train", gates[gates["reader_split"] == "train"]), ("test", gates[gates["reader_split"] == "test"])]:
        for expert, group in subset.groupby("dominant_expert"):
            row = {
                "run_dir": str(run_dir),
                "model": model_name,
                "seed": seed,
                "subset": split_name,
                "dominant_expert": int(expert),
                "n": int(len(group)),
                "mean_gate_entropy_norm": float(group["gate_entropy_norm"].mean()),
                "mean_dominant_weight": float(group["dominant_weight"].mean()),
            }
            for col in gate_cols:
                row[f"mean_{col}"] = float(group[col].mean())
            for col in behavior_cols:
                row[f"mean_{col}"] = float(group[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def compute_gate_stability(
    model: XLMRobertaForBehaviorConditionedTRT,
    data: pd.DataFrame,
    profile_stats,
    profile_feature_set: str,
    readers: list[str],
    train_readers: set[str],
    device: torch.device,
    run_dir: Path,
) -> pd.DataFrame:
    gates_by_label = {}
    for label, trials in [("trial1", [1]), ("trial2", [2]), ("trial1_2", [1, 2])]:
        raw_profiles = compute_behavior_profiles(data, profile_trials=trials, profile_feature_set=profile_feature_set)
        raw_profiles = raw_profiles.reindex(columns=profile_stats.feature_names).fillna(0.0)
        normalized = normalize_profiles(raw_profiles, profile_stats)
        gates_by_label[label] = compute_gates(model, [normalized[reader] for reader in readers], device=device)

    rows = []
    model_name, seed = parse_model_seed(run_dir.name)
    trial2_gates = gates_by_label["trial2"]
    for idx, reader in enumerate(readers):
        same_cosine = cosine(gates_by_label["trial1"][idx], gates_by_label["trial2"][idx])
        different = [
            cosine(gates_by_label["trial1"][idx], trial2_gates[other_idx])
            for other_idx, other_reader in enumerate(readers)
            if other_reader != reader
        ]
        different_l2 = [
            float(np.linalg.norm(gates_by_label["trial1"][idx] - trial2_gates[other_idx]))
            for other_idx, other_reader in enumerate(readers)
            if other_reader != reader
        ]
        same_l2 = float(np.linalg.norm(gates_by_label["trial1"][idx] - gates_by_label["trial2"][idx]))
        rows.append(
            {
                "run_dir": str(run_dir),
                "model": model_name,
                "seed": seed,
                "reader_id": reader,
                "reader_split": "train" if reader in train_readers else "test",
                "same_reader_cosine_t1_t2": float(same_cosine),
                "same_reader_l2_t1_t2": same_l2,
                "different_reader_cosine_mean": float(np.mean(different)),
                "different_reader_cosine_p95": float(np.quantile(different, 0.95)),
                "same_minus_different_cosine": float(same_cosine - np.mean(different)),
                "different_reader_l2_mean": float(np.mean(different_l2)),
                "different_reader_l2_p05": float(np.quantile(different_l2, 0.05)),
                "different_minus_same_l2": float(np.mean(different_l2) - same_l2),
            }
        )
    return pd.DataFrame(rows)


def load_reader_gains(run_dir: Path) -> pd.DataFrame | None:
    predictions_dir = run_dir / "predictions"
    actual_path = predictions_dir / "test_actual_predictions.csv"
    mean_path = predictions_dir / "test_mean_predictions.csv"
    shuffled_path = predictions_dir / "test_shuffled_predictions.csv"
    if not (actual_path.exists() and mean_path.exists() and shuffled_path.exists()):
        return None
    actual = pd.read_csv(actual_path).groupby("reader_id", as_index=False)["abs_error"].mean().rename(columns={"abs_error": "mae_actual"})
    mean = pd.read_csv(mean_path).groupby("reader_id", as_index=False)["abs_error"].mean().rename(columns={"abs_error": "mae_mean"})
    shuffled = pd.read_csv(shuffled_path).groupby("reader_id", as_index=False)["abs_error"].mean().rename(columns={"abs_error": "mae_shuffled"})
    gains = actual.merge(mean, on="reader_id").merge(shuffled, on="reader_id")
    gains["gain_vs_mean"] = gains["mae_mean"] - gains["mae_actual"]
    gains["gain_vs_shuffled"] = gains["mae_shuffled"] - gains["mae_actual"]

    model_name, seed = parse_model_seed(run_dir.name)
    concat_dir = run_dir.parent / f"concat_seed{seed}"
    concat_actual_path = concat_dir / "predictions" / "test_actual_predictions.csv"
    if concat_actual_path.exists():
        concat = (
            pd.read_csv(concat_actual_path)
            .groupby("reader_id", as_index=False)["abs_error"]
            .mean()
            .rename(columns={"abs_error": "concat_mae_actual"})
        )
        gains = gains.merge(concat, on="reader_id", how="left")
        gains["moe_gain_over_concat"] = gains["concat_mae_actual"] - gains["mae_actual"]
    return gains


def compute_gate_gain_correlations(gate_gains: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    gate_cols = gate_weight_cols(gate_gains)
    gate_features = gate_cols + ["gate_entropy_norm", "dominant_weight"]
    gain_cols = [col for col in ["gain_vs_mean", "gain_vs_shuffled", "moe_gain_over_concat"] if col in gate_gains.columns]
    rows = []
    model_name, seed = parse_model_seed(run_dir.name)
    for split_name, subset in [("all", gate_gains), ("test", gate_gains[gate_gains["reader_split"] == "test"])]:
        for gate_feature in gate_features:
            for gain_col in gain_cols:
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "model": model_name,
                        "seed": seed,
                        "subset": split_name,
                        "gate_feature": gate_feature,
                        "gain_metric": gain_col,
                        "n": int(len(subset)),
                        "pearson": safe_corr(subset[gate_feature], subset[gain_col]),
                        "spearman": safe_corr(subset[gate_feature].rank(), subset[gain_col].rank()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_dominant_expert_gains(gate_gains: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    gain_cols = [col for col in ["mae_actual", "gain_vs_mean", "gain_vs_shuffled", "moe_gain_over_concat"] if col in gate_gains.columns]
    rows = []
    model_name, seed = parse_model_seed(run_dir.name)
    for split_name, subset in [("all", gate_gains), ("test", gate_gains[gate_gains["reader_split"] == "test"])]:
        for expert, group in subset.groupby("dominant_expert"):
            row = {
                "run_dir": str(run_dir),
                "model": model_name,
                "seed": seed,
                "subset": split_name,
                "dominant_expert": int(expert),
                "n": int(len(group)),
                "mean_gate_entropy_norm": float(group["gate_entropy_norm"].mean()),
                "mean_dominant_weight": float(group["dominant_weight"].mean()),
            }
            for col in gain_cols:
                row[f"mean_{col}"] = float(group[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_runs(
    stability: pd.DataFrame,
    correlations: pd.DataFrame,
    gain_corr_frames: list[pd.DataFrame],
) -> dict[str, float | int | list[dict[str, float | int | str]]]:
    report = {
        "num_runs": int(stability["run_dir"].nunique()),
        "num_reader_rows": int(len(stability)),
        "mean_same_reader_trial1_trial2_cosine": float(stability["same_reader_cosine_t1_t2"].mean()),
        "mean_different_reader_cosine": float(stability["different_reader_cosine_mean"].mean()),
        "mean_same_minus_different_cosine": float(stability["same_minus_different_cosine"].mean()),
        "same_reader_beats_different_rate": float((stability["same_minus_different_cosine"] > 0).mean()),
        "mean_same_reader_trial1_trial2_l2": float(stability["same_reader_l2_t1_t2"].mean()),
        "mean_different_reader_l2": float(stability["different_reader_l2_mean"].mean()),
        "mean_different_minus_same_l2": float(stability["different_minus_same_l2"].mean()),
        "same_reader_l2_below_different_rate": float((stability["different_minus_same_l2"] > 0).mean()),
        "top_abs_gate_behavior_correlations": top_abs_correlations(correlations, top_k=20),
    }
    if gain_corr_frames:
        gain_corr = pd.concat(gain_corr_frames, ignore_index=True)
        entropy_gain = gain_corr[
            (gain_corr["subset"] == "test")
            & (gain_corr["gate_feature"] == "gate_entropy_norm")
            & (gain_corr["gain_metric"].isin(["gain_vs_mean", "gain_vs_shuffled", "moe_gain_over_concat"]))
        ]
        for _, row in entropy_gain.iterrows():
            key = f"entropy_spearman_{row['gain_metric']}_seed{row['seed']}"
            report[key] = float(row["spearman"])
    return report


def top_abs_correlations(correlations: pd.DataFrame, top_k: int) -> list[dict[str, float | int | str]]:
    subset = correlations.loc[correlations["subset"] == "all"].copy()
    subset["abs_spearman"] = subset["spearman"].abs()
    subset = subset.sort_values("abs_spearman", ascending=False).head(top_k)
    keep = ["model", "seed", "gate", "behavior_feature", "n", "pearson", "spearman"]
    return [
        {
            key: (None if pd.isna(value) else float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else str(value))
            for key, value in row.items()
        }
        for row in subset[keep].to_dict(orient="records")
    ]


def gate_weight_cols(data: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in data.columns if re.fullmatch(r"gate_\d+", col)],
        key=lambda name: int(name.split("_")[1]),
    )


def discover_moe_run_dirs(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and path.name.startswith("moe_seed")
        and (path / "best_model").exists()
        and (path / "data_summary.json").exists()
    )


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def parse_model_seed(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


def entropy(values: np.ndarray) -> float:
    clipped = np.clip(values, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(x.corr(y))


if __name__ == "__main__":
    main()
