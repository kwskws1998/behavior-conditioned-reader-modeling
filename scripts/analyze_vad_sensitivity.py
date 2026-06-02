#!/usr/bin/env python
"""Analyze reader-specific VAD sensitivity and VAD-conditioned prediction gains."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import (
    attach_vad_features,
    ensure_log_trt_column,
    load_vad_features,
    load_meco_rda,
    parse_int_list,
)


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"
PREDICTORS = [
    "vad_word_valence_centered",
    "vad_word_arousal_centered",
    "vad_word_dominance_centered",
    "vad_word_affective_intensity_centered",
]
OUTCOMES = {
    "log_trt": "log_trt",
    "nfix": "nfix",
    "firstrun_log_dur": "firstrun_log_dur",
    "reread": "reread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--vad-features-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--profile-trials", type=str, default="1,2")
    parser.add_argument("--future-trials", type=str, default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--high-quantile", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rda_path = resolve_rda_path(args.rda_path)
    output_dir = args.output_dir or ((args.run_root / "vad_sensitivity") if args.run_root else Path("artifacts/vad/vad_sensitivity"))
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_meco_rda(rda_path, lang=args.lang)
    data = attach_vad_features(data, load_vad_features(args.vad_features_path))
    data = prepare_analysis_columns(data)

    profile_trials = parse_int_list(args.profile_trials)
    future_trials = parse_int_list(args.future_trials)
    coefficients = build_coefficients(data, profile_trials=profile_trials, future_trials=future_trials)
    stability = summarize_stability(coefficients)
    prediction_rows = predict_future_from_calibration(data, profile_trials=profile_trials, future_trials=future_trials)
    prediction_summary = summarize_prediction_rows(prediction_rows)
    slice_summary = summarize_high_affect_slices(prediction_rows, high_quantile=args.high_quantile)

    coefficients.to_csv(output_dir / "vad_sensitivity_reader_coefficients.csv", index=False)
    stability.to_csv(output_dir / "vad_sensitivity_stability.csv", index=False)
    prediction_summary.to_csv(output_dir / "vad_sensitivity_prediction_summary.csv", index=False)
    slice_summary.to_csv(output_dir / "vad_sensitivity_slice_summary.csv", index=False)

    run_summary = pd.DataFrame()
    if args.run_root is not None:
        run_summary = summarize_run_root_predictions(
            run_root=args.run_root,
            vad_features=load_vad_features(args.vad_features_path),
            split=args.split,
            high_quantile=args.high_quantile,
        )
        run_summary.to_csv(output_dir / f"{args.split}_vad_prediction_gain_summary.csv", index=False)

    report = {
        "rda_path": str(rda_path),
        "vad_features_path": str(args.vad_features_path),
        "run_root": str(args.run_root) if args.run_root else None,
        "output_dir": str(output_dir),
        "profile_trials": profile_trials,
        "future_trials": future_trials,
        "split": args.split,
        "n_reader_coefficients": int(len(coefficients)),
        "n_prediction_rows": int(len(prediction_rows)),
        "n_run_prediction_rows": int(len(run_summary)),
        "top_stability": top_rows(stability, "calibration_future_pearson", 12),
        "prediction_summary": prediction_summary.to_dict(orient="records"),
    }
    (output_dir / "vad_sensitivity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def prepare_analysis_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = ensure_log_trt_column(data)
    data = data.copy()
    firstrun_duration = (
        pd.to_numeric(data["firstrun.dur"], errors="coerce")
        if "firstrun.dur" in data.columns
        else pd.Series(np.nan, index=data.index)
    )
    data["firstrun_log_dur"] = [
        None if pd.isna(value) else float(np.log1p(value))
        for value in firstrun_duration
    ]
    for col in [*PREDICTORS, "vad_word_arousal", "vad_word_affective_intensity", *OUTCOMES.values()]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def build_coefficients(data: pd.DataFrame, profile_trials: list[int], future_trials: list[int]) -> pd.DataFrame:
    rows = []
    for split_name, trials in [("calibration", profile_trials), ("future", future_trials)]:
        subset = data.loc[data["trialid"].isin(trials)].copy()
        for reader, group in subset.groupby("uniform_id", sort=True):
            for outcome_name, outcome_col in OUTCOMES.items():
                coeffs = fit_linear_coefficients(group, outcome_col)
                row_base = {
                    "split": split_name,
                    "reader_id": str(reader),
                    "outcome": outcome_name,
                    "n": int(coeffs["n"]),
                    "intercept": float(coeffs["intercept"]),
                }
                for predictor in PREDICTORS:
                    rows.append(
                        {
                            **row_base,
                            "predictor": predictor,
                            "slope": float(coeffs[predictor]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    calibration = coefficients.loc[coefficients["split"] == "calibration"].rename(columns={"slope": "calibration_slope"})
    future = coefficients.loc[coefficients["split"] == "future"].rename(columns={"slope": "future_slope"})
    merged = calibration.merge(
        future[["reader_id", "outcome", "predictor", "future_slope"]],
        on=["reader_id", "outcome", "predictor"],
        how="inner",
    )
    rows = []
    for (outcome, predictor), group in merged.groupby(["outcome", "predictor"], sort=True):
        rows.append(
            {
                "outcome": outcome,
                "predictor": predictor,
                "n_readers": int(group["reader_id"].nunique()),
                "calibration_slope_std": float(group["calibration_slope"].std(ddof=0)),
                "future_slope_std": float(group["future_slope"].std(ddof=0)),
                "calibration_future_pearson": safe_corr(group["calibration_slope"], group["future_slope"], method="pearson"),
                "calibration_future_spearman": safe_corr(group["calibration_slope"], group["future_slope"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def predict_future_from_calibration(data: pd.DataFrame, profile_trials: list[int], future_trials: list[int]) -> pd.DataFrame:
    calibration = data.loc[data["trialid"].isin(profile_trials)].copy()
    future = data.loc[data["trialid"].isin(future_trials)].copy()
    readers = sorted(data["uniform_id"].astype(str).unique())
    shuffled_readers = readers[1:] + readers[:1]
    shuffle_lookup = dict(zip(readers, shuffled_readers))
    rows = []
    for outcome_name, outcome_col in OUTCOMES.items():
        population = fit_linear_coefficients(calibration, outcome_col)
        reader_coeffs = {
            str(reader): fit_linear_coefficients(group, outcome_col)
            for reader, group in calibration.groupby("uniform_id", sort=True)
        }
        usable = future[
            [outcome_col, "uniform_id", "trialid", "sentnum", "wordnum", "vad_word_arousal", "vad_word_affective_intensity", *PREDICTORS]
        ].copy()
        numeric_cols = [outcome_col, "trialid", "sentnum", "wordnum", "vad_word_arousal", "vad_word_affective_intensity", *PREDICTORS]
        for col in numeric_cols:
            usable[col] = pd.to_numeric(usable[col], errors="coerce")
        usable = usable.replace([np.inf, -np.inf], np.nan)
        usable = usable.dropna(subset=[outcome_col, *PREDICTORS]).copy()
        for _, item in usable.iterrows():
            reader = str(item["uniform_id"])
            shuffled_reader = shuffle_lookup.get(reader, reader)
            y = float(item[outcome_col])
            actual_pred = predict_from_coefficients(item, reader_coeffs.get(reader, population))
            population_pred = predict_from_coefficients(item, population)
            shuffled_pred = predict_from_coefficients(item, reader_coeffs.get(shuffled_reader, population))
            rows.append(
                {
                    "outcome": outcome_name,
                    "reader_id": reader,
                    "trialid": int(item["trialid"]),
                    "sentnum": int(item["sentnum"]),
                    "wordnum": int(item["wordnum"]),
                    "true_value": y,
                    "pred_actual_reader": actual_pred,
                    "pred_population": population_pred,
                    "pred_shuffled_reader": shuffled_pred,
                    "abs_error_actual_reader": abs(y - actual_pred),
                    "abs_error_population": abs(y - population_pred),
                    "abs_error_shuffled_reader": abs(y - shuffled_pred),
                    "gain_vs_population": abs(y - population_pred) - abs(y - actual_pred),
                    "gain_vs_shuffled": abs(y - shuffled_pred) - abs(y - actual_pred),
                    "vad_word_arousal": float(item["vad_word_arousal"]),
                    "vad_word_affective_intensity": float(item["vad_word_affective_intensity"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for outcome, group in rows.groupby("outcome", sort=True):
        summary.append(
            {
                "outcome": outcome,
                "n": int(len(group)),
                "mae_actual_reader": float(group["abs_error_actual_reader"].mean()),
                "mae_population": float(group["abs_error_population"].mean()),
                "mae_shuffled_reader": float(group["abs_error_shuffled_reader"].mean()),
                "gain_vs_population": float(group["gain_vs_population"].mean()),
                "gain_vs_shuffled": float(group["gain_vs_shuffled"].mean()),
            }
        )
    return pd.DataFrame(summary)


def summarize_high_affect_slices(rows: pd.DataFrame, high_quantile: float) -> pd.DataFrame:
    output = []
    for outcome, group in rows.groupby("outcome", sort=True):
        for slice_name, col in [
            ("high_arousal", "vad_word_arousal"),
            ("high_affective_intensity", "vad_word_affective_intensity"),
        ]:
            threshold = group[col].quantile(high_quantile)
            for label, subset in [("all", group), (slice_name, group.loc[group[col] >= threshold])]:
                output.append(
                    {
                        "outcome": outcome,
                        "slice": label,
                        "threshold_feature": col,
                        "threshold": float(threshold),
                        "n": int(len(subset)),
                        "gain_vs_population": float(subset["gain_vs_population"].mean()),
                        "gain_vs_shuffled": float(subset["gain_vs_shuffled"].mean()),
                    }
                )
    return pd.DataFrame(output)


def summarize_run_root_predictions(run_root: Path, vad_features: pd.DataFrame, split: str, high_quantile: float) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(run_root):
        predictions_dir = run_dir / "predictions"
        files = {
            mode: predictions_dir / f"{split}_{mode}_predictions.csv"
            for mode in ["actual", "mean", "shuffled"]
        }
        if not all(path.exists() for path in files.values()):
            continue
        merged = merge_prediction_modes(files)
        merged = attach_prediction_vad(merged, vad_features)
        model, seed = parse_model_seed(run_dir.name)
        error_col = "residual_abs_error" if "residual_abs_error_actual" in merged.columns else "abs_error"
        rows.extend(summarize_prediction_gain_slices(merged, run_dir, model, seed, error_col, high_quantile))
    return pd.DataFrame(rows)


def discover_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and (path / "data_summary.json").exists() and (path / "predictions").exists()
    )


def merge_prediction_modes(files: dict[str, Path]) -> pd.DataFrame:
    frames = []
    keys = ["reader_id", "trial_id", "sent_num", "word_num", "word_idx", "word", "true_log_trt"]
    for mode, path in files.items():
        frame = pd.read_csv(path)
        keep = [col for col in keys if col in frame.columns]
        value_cols = [
            col
            for col in ["abs_error", "residual_abs_error", "reconstructed_abs_error", "pred_log_trt", "pred_residual_log_trt"]
            if col in frame.columns
        ]
        frame = frame[[*keep, *value_cols]].copy()
        frame = frame.rename(columns={col: f"{col}_{mode}" for col in value_cols})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=keys, how="inner")
    return merged


def attach_prediction_vad(predictions: pd.DataFrame, vad_features: pd.DataFrame) -> pd.DataFrame:
    vad = vad_features.rename(
        columns={
            "uniform_id": "reader_id",
            "trialid": "trial_id",
            "sentnum": "sent_num",
            "wordnum": "word_num",
        }
    )
    return predictions.merge(
        vad[["reader_id", "trial_id", "sent_num", "word_num", "vad_word_arousal", "vad_word_affective_intensity"]],
        on=["reader_id", "trial_id", "sent_num", "word_num"],
        how="left",
        validate="many_to_one",
    )


def summarize_prediction_gain_slices(
    data: pd.DataFrame,
    run_dir: Path,
    model: str,
    seed: int | None,
    error_col: str,
    high_quantile: float,
) -> list[dict[str, object]]:
    actual_col = f"{error_col}_actual"
    mean_col = f"{error_col}_mean"
    shuffled_col = f"{error_col}_shuffled"
    data = data.dropna(subset=[actual_col, mean_col, shuffled_col]).copy()
    data["gain_vs_mean"] = data[mean_col] - data[actual_col]
    data["gain_vs_shuffled"] = data[shuffled_col] - data[actual_col]
    rows = []
    for slice_name, col in [
        ("all", None),
        ("high_arousal", "vad_word_arousal"),
        ("high_affective_intensity", "vad_word_affective_intensity"),
    ]:
        if col is None:
            subset = data
            threshold = np.nan
        else:
            threshold = data[col].quantile(high_quantile)
            subset = data.loc[data[col] >= threshold]
        rows.append(
            {
                "run_dir": str(run_dir),
                "model": model,
                "seed": seed,
                "slice": slice_name,
                "threshold": float(threshold) if np.isfinite(threshold) else None,
                "n": int(len(subset)),
                "mae_actual": float(subset[actual_col].mean()),
                "mae_mean": float(subset[mean_col].mean()),
                "mae_shuffled": float(subset[shuffled_col].mean()),
                "gain_vs_mean": float(subset["gain_vs_mean"].mean()),
                "gain_vs_shuffled": float(subset["gain_vs_shuffled"].mean()),
            }
        )
    return rows


def fit_linear_coefficients(data: pd.DataFrame, outcome_col: str) -> dict[str, float]:
    subset = data[[outcome_col, *PREDICTORS]].apply(pd.to_numeric, errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    result = {"n": float(len(subset)), "intercept": 0.0}
    result.update({predictor: 0.0 for predictor in PREDICTORS})
    if len(subset) < len(PREDICTORS) + 2:
        return result
    x = subset[PREDICTORS].to_numpy(dtype=float)
    y = subset[outcome_col].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return result
    result["intercept"] = float(beta[0])
    for predictor, value in zip(PREDICTORS, beta[1:]):
        result[predictor] = float(value)
    return result


def predict_from_coefficients(row: pd.Series, coeffs: dict[str, float]) -> float:
    value = float(coeffs["intercept"])
    for predictor in PREDICTORS:
        value += float(coeffs[predictor]) * float(row[predictor])
    return value


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def top_rows(data: pd.DataFrame, column: str, n: int) -> list[dict[str, object]]:
    if data.empty or column not in data.columns:
        return []
    ranked = data.assign(abs_value=data[column].abs()).sort_values("abs_value", ascending=False)
    return ranked.drop(columns=["abs_value"]).head(n).to_dict(orient="records")


def parse_model_seed(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_seed(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


if __name__ == "__main__":
    main()
