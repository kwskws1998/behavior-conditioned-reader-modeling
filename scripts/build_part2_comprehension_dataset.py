#!/usr/bin/env python
"""Build trial-level Part 2 features for comprehension-risk modeling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import compute_behavior_profiles, fit_profile_stats, load_meco_rda, normalize_profiles


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"
DEFAULT_CORRECTNESS_CANDIDATES = [
    "correct",
    "is_correct",
    "answer_correct",
    "question_correct",
    "q_correct",
    "qcorrect",
    "accuracy",
    "acc",
    "comp_acc",
    "comprehension",
    "score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--part1-run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--correctness-column", type=str, default="")
    parser.add_argument("--question-column", type=str, default="")
    parser.add_argument("--profile-trials", type=str, default="")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rda_path = resolve_rda_path(args.rda_path)
    summary = json.loads((args.part1_run_dir / "data_summary.json").read_text(encoding="utf-8"))
    data = load_meco_rda(rda_path, lang=args.lang)
    correctness_col = args.correctness_column or infer_correctness_column(data)
    question_col = args.question_column or infer_question_column(data)
    profile_trials = parse_trials(args.profile_trials) if args.profile_trials else summary.get("profile_trials", [1, 2])

    labels = build_correctness_labels(
        data=data,
        correctness_col=correctness_col,
        question_col=question_col,
        positive_threshold=args.positive_threshold,
    )
    if labels.empty:
        raise ValueError(f"No labels produced from correctness column {correctness_col!r}")

    text_features = build_text_features(data, question_col=question_col)
    profile_features = build_profile_features(data, summary=summary, profile_trials=profile_trials)
    split_features = build_split_features(summary)
    gaze_features = build_gaze_features(args.part1_run_dir)

    dataset = labels.merge(text_features, on=example_keys(question_col), how="left")
    dataset = dataset.merge(profile_features, on="reader_id", how="left")
    dataset = dataset.merge(split_features, on=["reader_id", "trial_id"], how="left")
    dataset = dataset.merge(gaze_features, on=["split", "reader_id", "trial_id"], how="left")
    dataset = dataset.dropna(subset=["split"]).copy()
    dataset = dataset.sort_values(["split", "reader_id", "trial_id"]).reset_index(drop=True)

    output_path = args.output_path or args.part1_run_dir / "part2_comprehension" / "part2_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    manifest = {
        "rda_path": str(rda_path),
        "part1_run_dir": str(args.part1_run_dir),
        "output_path": str(output_path),
        "correctness_column": correctness_col,
        "question_column": question_col,
        "profile_trials": profile_trials,
        "n_rows": int(len(dataset)),
        "label_mean": float(dataset["correct"].mean()) if len(dataset) else None,
        "split_counts": dataset["split"].value_counts().to_dict(),
        "available_gaze_feature_sets": sorted(
            {
                col.removeprefix("gaze_").split("_", 1)[0]
                for col in dataset.columns
                if col.startswith("gaze_")
            }
        ),
    }
    (output_path.parent / "part2_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def parse_trials(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_correctness_column(data: pd.DataFrame) -> str:
    lowered = {str(col).lower(): str(col) for col in data.columns}
    for candidate in DEFAULT_CORRECTNESS_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    keyword_hits = [
        str(col)
        for col in data.columns
        if any(keyword in str(col).lower() for keyword in ["correct", "accuracy", "compr", "score"])
    ]
    if len(keyword_hits) == 1:
        return keyword_hits[0]
    raise ValueError(
        "Could not infer correctness column. Run scripts/inspect_meco_schema.py and pass "
        "--correctness-column explicitly. Candidates: "
        + ", ".join(keyword_hits[:20])
    )


def infer_question_column(data: pd.DataFrame) -> str:
    for candidate in ["question_id", "qid", "q_id", "question", "qnum", "itemid", "item_id"]:
        for col in data.columns:
            if str(col).lower() == candidate:
                return str(col)
    return ""


def example_keys(question_col: str) -> list[str]:
    keys = ["reader_id", "trial_id"]
    if question_col:
        keys.append("question_id")
    return keys


def build_correctness_labels(
    data: pd.DataFrame,
    correctness_col: str,
    question_col: str,
    positive_threshold: float,
) -> pd.DataFrame:
    subset = data.copy()
    subset["reader_id"] = subset["uniform_id"].astype(str)
    subset["trial_id"] = subset["trialid"].astype(int)
    subset["correct_value"] = pd.to_numeric(subset[correctness_col], errors="coerce")
    subset = subset.dropna(subset=["correct_value"])
    if question_col:
        subset["question_id"] = subset[question_col].astype(str)
    keys = example_keys(question_col)
    labels = subset.groupby(keys, as_index=False)["correct_value"].mean()
    labels["correct"] = (labels["correct_value"] >= positive_threshold).astype(int)
    return labels.drop(columns=["correct_value"])


def build_text_features(data: pd.DataFrame, question_col: str) -> pd.DataFrame:
    subset = data.copy()
    subset["reader_id"] = subset["uniform_id"].astype(str)
    subset["trial_id"] = subset["trialid"].astype(int)
    subset["word_str"] = subset["word"].astype(str)
    subset["word_len"] = subset["word_str"].str.len()
    subset["is_punct"] = subset["word_str"].str.fullmatch(r"\W+").fillna(False).astype(float)
    if question_col:
        subset["question_id"] = subset[question_col].astype(str)
    rows = []
    for keys, group in subset.groupby(example_keys(question_col), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(example_keys(question_col), keys))
        words = group.sort_values(["sentnum", "wordnum"])["word_str"].tolist()
        word_lengths = group["word_len"].to_numpy(dtype=float)
        row.update(
            {
                "text": " ".join(words),
                "text_n_words": int(len(words)),
                "text_n_sentences": int(group["sentnum"].nunique()),
                "text_mean_word_len": float(np.nanmean(word_lengths)),
                "text_std_word_len": float(np.nanstd(word_lengths)),
                "text_max_word_len": float(np.nanmax(word_lengths)),
                "text_unique_ratio": float(len(set(words)) / max(len(words), 1)),
                "text_punct_rate": float(group["is_punct"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_profile_features(data: pd.DataFrame, summary: dict, profile_trials: list[int]) -> pd.DataFrame:
    raw_profiles = compute_behavior_profiles(data, profile_trials=profile_trials)
    stats = fit_profile_stats(raw_profiles, summary["train_readers"])
    normalized = normalize_profiles(raw_profiles, stats)
    rows = []
    for reader, values in normalized.items():
        row = {"reader_id": reader}
        for feature_name, value in zip(stats.feature_names, values):
            row[f"profile_{feature_name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_split_features(summary: dict) -> pd.DataFrame:
    train_readers = set(summary["train_readers"])
    test_readers = set(summary["test_readers"])
    train_trials = set(summary.get("train_trials", [3, 4, 5, 6, 7, 8, 9, 10]))
    dev_trials = set(summary.get("dev_trials", [11, 12]))
    rows = []
    for reader in sorted(train_readers):
        for trial in train_trials:
            rows.append({"reader_id": reader, "trial_id": int(trial), "split": "train"})
        for trial in dev_trials:
            rows.append({"reader_id": reader, "trial_id": int(trial), "split": "dev"})
    for reader in sorted(test_readers):
        for trial in dev_trials:
            rows.append({"reader_id": reader, "trial_id": int(trial), "split": "test"})
    return pd.DataFrame(rows)


def build_gaze_features(run_dir: Path) -> pd.DataFrame:
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Missing predictions directory: {predictions_dir}")
    frames = []
    for path in sorted(predictions_dir.glob("*_predictions.csv")):
        split, profile_mode = parse_prediction_name(path.name)
        if split not in {"train", "dev", "test"}:
            continue
        frame = summarize_prediction_file(path, split=split, profile_mode=profile_mode)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction CSV files found under {predictions_dir}")
    merged = None
    for frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["split", "reader_id", "trial_id"], how="outer")
    return merged


def parse_prediction_name(name: str) -> tuple[str, str]:
    stem = name.removesuffix("_predictions.csv")
    pieces = stem.split("_")
    split = pieces[0]
    profile_mode = "_".join(pieces[1:])
    return split, profile_mode


def summarize_prediction_file(path: Path, split: str, profile_mode: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    rows = []
    prefix = f"gaze_{profile_mode}"
    for (reader, trial), group in data.groupby(["reader_id", "trial_id"], sort=True):
        pred = group["pred_log_trt"].to_numpy(dtype=float)
        row = {
            "split": split,
            "reader_id": str(reader),
            "trial_id": int(trial),
            f"{prefix}_mean": float(np.mean(pred)),
            f"{prefix}_std": float(np.std(pred)),
            f"{prefix}_sum": float(np.sum(pred)),
            f"{prefix}_max": float(np.max(pred)),
            f"{prefix}_p90": float(np.quantile(pred, 0.90)),
            f"{prefix}_top10_mean": float(np.mean(pred[pred >= np.quantile(pred, 0.90)])),
        }
        if "abs_error" in group.columns:
            row[f"{prefix}_mae_observed_only"] = float(group["abs_error"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
