#!/usr/bin/env python
"""Train Part 2 comprehension-risk models from behavior and predicted gaze features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-column", type=str, default="correct")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.dataset_path)
    output_dir = args.output_dir or args.dataset_path.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_feature_sets(data)
    train = data[data["split"] == "train"].copy()
    dev = data[data["split"] == "dev"].copy()
    test = data[data["split"] == "test"].copy()
    if train.empty:
        raise ValueError("Part 2 dataset has no train split. Check data_summary split fields.")

    results = []
    predictions = []
    coefficients = []
    skipped = []
    for model_name, features in feature_sets.items():
        available_features = [feature for feature in features if feature in data.columns]
        skip_reason = get_skip_reason(model_name, available_features, train, args.target_column)
        if skip_reason:
            skipped.append({"model": model_name, "reason": skip_reason, "n_available_features": len(available_features)})
            continue
        model = make_model(model_name=model_name, seed=args.seed, max_iter=args.max_iter)
        model.fit(train[available_features], train[args.target_column].astype(int))
        for split_name, split_data in [("train", train), ("dev", dev), ("test", test)]:
            if split_data.empty:
                continue
            y_true = split_data[args.target_column].astype(int).to_numpy()
            y_prob = predict_probability(model, split_data[available_features])
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = compute_metrics(y_true, y_prob, y_pred)
            results.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "n": int(len(split_data)),
                    "positive_rate": float(y_true.mean()),
                    "n_features": int(len(available_features)),
                    **metrics,
                }
            )
            split_predictions = split_data[["reader_id", "trial_id", "split", args.target_column]].copy()
            if "question_id" in split_data.columns:
                split_predictions["question_id"] = split_data["question_id"]
            split_predictions["model"] = model_name
            split_predictions["prob_correct"] = y_prob
            split_predictions["pred_correct"] = y_pred
            predictions.append(split_predictions)
        coefficients.extend(extract_coefficients(model, model_name, available_features))

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    coefficients_df = pd.DataFrame(coefficients)
    skipped_df = pd.DataFrame(skipped)
    results_df.to_csv(output_dir / "part2_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "part2_predictions.csv", index=False)
    coefficients_df.to_csv(output_dir / "part2_coefficients.csv", index=False)
    skipped_df.to_csv(output_dir / "part2_skipped_models.csv", index=False)

    summary = summarize_results(results_df, skipped_df)
    (output_dir / "part2_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    data["constant_1"] = 1.0
    question = sorted(col for col in data.columns if col == "question_num" or col.startswith("question_is_"))
    item = sorted(
        col
        for col in data.columns
        if col == "trial_num" or col.startswith("trial_is_") or col.startswith("trial_")
    )
    text = sorted(col for col in data.columns if col.startswith("text_"))
    profile = sorted(col for col in data.columns if col.startswith("profile_"))
    gaze_actual = sorted(col for col in data.columns if col.startswith("gaze_actual_") and not col.endswith("_observed_only"))
    gaze_mean = sorted(col for col in data.columns if col.startswith("gaze_mean_") and not col.endswith("_observed_only"))
    gaze_shuffled = sorted(col for col in data.columns if col.startswith("gaze_shuffled_") and not col.endswith("_observed_only"))

    interactions = make_interaction_features(data, profile, gaze_actual)
    return {
        "majority": ["constant_1"],
        "item_question_only": item + question,
        "text_only": question + text,
        "mf_like_behavior": question + text + profile,
        "mb_like_predicted_gaze": question + text + gaze_actual,
        "mean_gaze": question + text + gaze_mean,
        "shuffled_gaze": question + text + gaze_shuffled,
        "arbitration_profile_x_gaze": question + text + profile + gaze_actual + interactions,
        "item_question_plus_profile": item + question + profile,
        "item_question_plus_predicted_gaze": item + question + gaze_actual,
        "item_question_plus_mean_gaze": item + question + gaze_mean,
        "item_question_plus_shuffled_gaze": item + question + gaze_shuffled,
    }


def make_interaction_features(data: pd.DataFrame, profile: list[str], gaze_actual: list[str]) -> list[str]:
    selected_profile = [
        col
        for col in profile
        if any(key in col for key in ["reread", "reg.in", "reg.out", "skip"])
    ]
    selected_gaze = [
        col
        for col in gaze_actual
        if any(key in col for key in ["mean", "p90", "top10_mean", "std"])
    ]
    interaction_values = {}
    for p_col in selected_profile:
        for g_col in selected_gaze:
            name = f"interaction__{p_col}__x__{g_col}"
            interaction_values[name] = data[p_col] * data[g_col]
    if interaction_values:
        interactions = pd.DataFrame(interaction_values, index=data.index)
        for col in interactions.columns:
            data[col] = interactions[col]
    return list(interaction_values)


def make_model(model_name: str, seed: int, max_iter: int):
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def has_required_signal(model_name: str, train: pd.DataFrame) -> bool:
    prefix = required_signal_prefix(model_name)
    if prefix is None:
        return True
    cols = [col for col in train.columns if col.startswith(prefix) and not col.endswith("_observed_only")]
    return bool(cols) and train[cols].notna().sum().sum() > 0


def required_signal_prefix(model_name: str) -> str | None:
    requirements = {
        "mb_like_predicted_gaze": "gaze_actual_",
        "mean_gaze": "gaze_mean_",
        "shuffled_gaze": "gaze_shuffled_",
        "arbitration_profile_x_gaze": "gaze_actual_",
        "item_question_plus_predicted_gaze": "gaze_actual_",
        "item_question_plus_mean_gaze": "gaze_mean_",
        "item_question_plus_shuffled_gaze": "gaze_shuffled_",
    }
    return requirements.get(model_name)


def get_skip_reason(model_name: str, available_features: list[str], train: pd.DataFrame, target_column: str) -> str:
    if not available_features:
        return "no available feature columns"
    if train[available_features].notna().sum().sum() == 0:
        return "all selected feature values are missing in train"
    prefix = required_signal_prefix(model_name)
    if prefix is not None:
        cols = [col for col in train.columns if col.startswith(prefix) and not col.endswith("_observed_only")]
        if not cols:
            return f"no columns with required prefix {prefix!r}"
        if train[cols].notna().sum().sum() == 0:
            return f"required prefix {prefix!r} exists, but all train values are missing"
    if model_name != "majority" and train[target_column].nunique() < 2:
        return "train split has fewer than two target classes"
    return ""


def predict_probability(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if probabilities.shape[1] == 1:
        return np.full(len(features), float(model.classes_[0]))
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, clipped))
    else:
        metrics["auroc"] = float("nan")
        metrics["average_precision"] = float("nan")
        metrics["log_loss"] = float("nan")
    return metrics


def extract_coefficients(model, model_name: str, features: list[str]) -> list[dict[str, object]]:
    if model_name == "majority" or not hasattr(model, "named_steps"):
        return []
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    coefs = classifier.coef_[0]
    rows = []
    for feature, coef in zip(features, coefs):
        rows.append({"model": model_name, "feature": feature, "coefficient": float(coef), "abs_coefficient": float(abs(coef))})
    return rows


def summarize_results(results: pd.DataFrame, skipped: pd.DataFrame) -> dict[str, object]:
    report = {
        "metrics_path": "part2_metrics.csv",
        "predictions_path": "part2_predictions.csv",
        "coefficients_path": "part2_coefficients.csv",
        "skipped_models_path": "part2_skipped_models.csv",
        "test_metrics": {},
        "main_comparisons": {},
        "skipped_models": skipped.to_dict(orient="records") if not skipped.empty else [],
    }
    test = results[results["split"] == "test"].copy()
    for _, row in test.iterrows():
        report["test_metrics"][row["model"]] = {
            key: float(row[key])
            for key in ["n", "positive_rate", "accuracy", "balanced_accuracy", "auroc", "average_precision", "brier", "log_loss"]
            if key in row and not pd.isna(row[key])
        }
    report["main_comparisons"] = compare_models(test, metric="balanced_accuracy")
    return report


def compare_models(test: pd.DataFrame, metric: str) -> dict[str, float]:
    values = {row["model"]: float(row[metric]) for _, row in test.iterrows() if not pd.isna(row[metric])}
    comparisons = {}
    pairs = [
        ("mb_like_predicted_gaze", "mf_like_behavior"),
        ("mb_like_predicted_gaze", "text_only"),
        ("item_question_plus_predicted_gaze", "item_question_only"),
        ("item_question_plus_predicted_gaze", "item_question_plus_mean_gaze"),
        ("item_question_plus_predicted_gaze", "item_question_plus_shuffled_gaze"),
        ("item_question_plus_predicted_gaze", "item_question_plus_profile"),
        ("arbitration_profile_x_gaze", "mb_like_predicted_gaze"),
        ("arbitration_profile_x_gaze", "mf_like_behavior"),
        ("mb_like_predicted_gaze", "mean_gaze"),
        ("mb_like_predicted_gaze", "shuffled_gaze"),
    ]
    for left, right in pairs:
        if left in values and right in values:
            comparisons[f"{left}_minus_{right}_{metric}"] = values[left] - values[right]
    return comparisons


if __name__ == "__main__":
    main()
