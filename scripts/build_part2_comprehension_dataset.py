#!/usr/bin/env python
"""Build question-level Part 2 features for comprehension-risk modeling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

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
LANGUAGE_NAMES = {
    "du": "Dutch",
    "ee": "Estonian",
    "en": "English",
    "fi": "Finnish",
    "ge": "German",
    "gr": "Greek",
    "he": "Hebrew",
    "it": "Italian",
    "ko": "Korean",
    "no": "Norwegian",
    "ru": "Russian",
    "sp": "Spanish",
    "tr": "Turkish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--comprehension-path", type=Path, default=None)
    parser.add_argument("--part1-run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--correctness-column", type=str, default="")
    parser.add_argument("--question-column", type=str, default="")
    parser.add_argument("--question-materials-path", type=Path, default=None)
    parser.add_argument("--text-materials-path", type=Path, default=None)
    parser.add_argument("--profile-trials", type=str, default="")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rda_path = resolve_rda_path(args.rda_path)
    summary = json.loads((args.part1_run_dir / "data_summary.json").read_text(encoding="utf-8"))
    data = load_meco_rda(rda_path, lang=args.lang)
    label_data = load_comprehension_data(args.comprehension_path, args.lang) if args.comprehension_path else data
    correctness_col = args.correctness_column or infer_correctness_column(label_data)
    question_col = args.question_column or infer_question_column(label_data)
    profile_trials = parse_trials(args.profile_trials) if args.profile_trials else summary.get("profile_trials", [1, 2])

    labels = build_correctness_labels(
        data=label_data,
        correctness_col=correctness_col,
        question_col=question_col,
        positive_threshold=args.positive_threshold,
    )
    if labels.empty:
        raise ValueError(f"No labels produced from correctness column {correctness_col!r}")

    text_features = build_text_features(data, question_col=question_col if question_col in data.columns else "")
    profile_features = build_profile_features(data, summary=summary, profile_trials=profile_trials)
    split_features = build_split_features(summary)
    gaze_features = build_gaze_features(args.part1_run_dir)

    dataset = labels.merge(text_features, on=["reader_id", "trial_id"], how="left")
    material_features = build_material_features(
        question_materials_path=args.question_materials_path,
        text_materials_path=args.text_materials_path,
        lang=args.lang,
        require_question_id="question_id" in labels.columns,
    )
    if not material_features.empty:
        material_keys = ["trial_id"]
        if "question_id" in labels.columns and "question_id" in material_features.columns:
            material_keys.append("question_id")
        dataset = dataset.merge(material_features, on=material_keys, how="left")
    dataset = dataset.merge(profile_features, on="reader_id", how="left")
    dataset = dataset.merge(split_features, on=["reader_id", "trial_id"], how="left")
    dataset = dataset.merge(gaze_features, on=["split", "reader_id", "trial_id"], how="left")
    dataset = add_question_features(dataset)
    dataset = compose_model_text(dataset)
    dataset = dataset.dropna(subset=["split"]).copy()
    dataset = dataset.sort_values(["split", "reader_id", "trial_id"]).reset_index(drop=True)

    output_path = args.output_path or args.part1_run_dir / "part2_comprehension" / "part2_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    manifest = {
        "rda_path": str(rda_path),
        "comprehension_path": str(args.comprehension_path) if args.comprehension_path else None,
        "part1_run_dir": str(args.part1_run_dir),
        "output_path": str(output_path),
        "correctness_column": correctness_col,
        "question_column": question_col,
        "question_materials_path": str(args.question_materials_path) if args.question_materials_path else None,
        "text_materials_path": str(args.text_materials_path) if args.text_materials_path else None,
        "profile_trials": profile_trials,
        "n_rows": int(len(dataset)),
        "n_with_question_text": int(dataset["question_text"].notna().sum()) if "question_text" in dataset.columns else 0,
        "n_with_material_passage_text": int(dataset["passage_text"].notna().sum()) if "passage_text" in dataset.columns else 0,
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


def load_comprehension_data(path: Path | None, lang: str) -> pd.DataFrame:
    if path is None:
        raise ValueError("comprehension path is required")
    if not path.exists():
        candidates = sorted(Path("data").rglob(path.name))
        if len(candidates) == 1:
            path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find comprehension file: {path}")
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No data frames found in {path}")
    data = next(iter(result.values()))
    if "lang" in data.columns:
        data = data.loc[data["lang"].astype(str) == lang].copy()
    return data.reset_index(drop=True)


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
    for candidate in ["QUESTIONNUM", "question_id", "qid", "q_id", "question", "qnum", "itemid", "item_id", "number"]:
        for col in data.columns:
            if str(col).lower() == candidate.lower():
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
    if "uniform_id" not in subset.columns:
        raise ValueError("Label data must contain uniform_id")
    if "trialid" not in subset.columns:
        raise ValueError("Label data must contain trialid")
    subset["reader_id"] = subset["uniform_id"].astype(str)
    subset["trial_id"] = pd.to_numeric(subset["trialid"], errors="coerce").astype("Int64")
    subset["correct_value"] = pd.to_numeric(subset[correctness_col], errors="coerce")
    subset = subset.dropna(subset=["trial_id", "correct_value"])
    subset["trial_id"] = subset["trial_id"].astype(int)
    if question_col:
        subset["question_id"] = normalize_question_ids(subset[question_col])
    keys = example_keys(question_col)
    labels = subset.groupby(keys, as_index=False)["correct_value"].mean()
    labels["correct"] = (labels["correct_value"] >= positive_threshold).astype(int)
    return labels.drop(columns=["correct_value"])


def normalize_question_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype(str).str.strip()
    integral = numeric.notna() & np.isclose(numeric, np.round(numeric))
    normalized.loc[integral] = numeric.loc[integral].round().astype(int).astype(str)
    return normalized


def add_question_features(data: pd.DataFrame) -> pd.DataFrame:
    if "question_id" not in data.columns:
        return data
    data = data.copy()
    question_num = pd.to_numeric(data["question_id"], errors="coerce")
    if question_num.notna().any():
        data["question_num"] = question_num.fillna(question_num.median()).astype(float)
        for value in sorted(int(v) for v in question_num.dropna().unique()):
            data[f"question_is_{value}"] = (question_num == value).astype(float)
    return data


def build_material_features(
    question_materials_path: Path | None,
    text_materials_path: Path | None,
    lang: str,
    require_question_id: bool,
) -> pd.DataFrame:
    frames = []
    if text_materials_path:
        frames.append(load_passage_materials(resolve_existing_path(text_materials_path), lang=lang))
    if question_materials_path:
        if not require_question_id:
            raise ValueError("--question-materials-path requires a question column in the label data")
        frames.append(load_question_materials(resolve_existing_path(question_materials_path), lang=lang))
    if not frames:
        return pd.DataFrame()
    material = frames[0]
    for frame in frames[1:]:
        keys = ["trial_id"]
        if "question_id" in material.columns and "question_id" in frame.columns:
            keys.append("question_id")
        material = material.merge(frame, on=keys, how="outer")
    return material


def resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(path.name))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find materials file: {path}")


def load_question_materials(path: Path, lang: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheet_name = find_question_sheet(xls.sheet_names, lang)
        table = pd.read_excel(path, sheet_name=sheet_name)
    else:
        table = pd.read_csv(path)
    if "number" not in table.columns:
        raise ValueError(f"Question materials must contain a number column: {path}")

    rows = []
    for _, source in table.iterrows():
        trial_id = pd.to_numeric(source["number"], errors="coerce")
        if pd.isna(trial_id):
            continue
        for idx in range(1, 5):
            q_col = f"q{idx}"
            a_col = f"a{idx}"
            if q_col not in table.columns:
                continue
            question_text = clean_text(source.get(q_col, ""))
            if not question_text:
                continue
            row = {
                "trial_id": int(trial_id),
                "question_id": str(idx),
                "question_text": question_text,
            }
            if a_col in table.columns:
                answer = pd.to_numeric(source.get(a_col), errors="coerce")
                if pd.notna(answer):
                    row["answer_key"] = int(answer)
            rows.append(row)
    return pd.DataFrame(rows)


def find_question_sheet(sheet_names: list[str], lang: str) -> str:
    lowered = {sheet.lower(): sheet for sheet in sheet_names}
    if lang.lower() in lowered:
        return lowered[lang.lower()]
    language_name = LANGUAGE_NAMES.get(lang.lower(), lang).lower()
    if language_name in lowered:
        return lowered[language_name]
    raise ValueError(f"Could not find sheet for language {lang!r}. Available sheets: {sheet_names}")


def load_passage_materials(path: Path, lang: str) -> pd.DataFrame:
    tables = read_material_tables(path)
    for table in tables:
        row = select_language_row(table, lang)
        if row is None:
            continue
        records = []
        trial_id = 1
        for col in table.columns[1:]:
            if str(col).lower().startswith("unnamed"):
                continue
            text = clean_text(row.get(col, ""))
            if not text:
                continue
            records.append(
                {
                    "trial_id": trial_id,
                    "passage_title": str(col),
                    "passage_text": text,
                }
            )
            trial_id += 1
        if records:
            return pd.DataFrame(records)
    raise ValueError(f"Could not find passage materials for language {lang!r} in {path}")


def read_material_tables(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        return [pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names]
    return [pd.read_csv(path)]


def select_language_row(table: pd.DataFrame, lang: str) -> pd.Series | None:
    if table.empty:
        return None
    first_col = table.columns[0]
    language_name = LANGUAGE_NAMES.get(lang.lower(), lang)
    wanted = {lang.lower(), language_name.lower()}
    labels = table[first_col].astype(str).str.strip().str.lower()
    matches = table.loc[labels.isin(wanted)]
    if matches.empty:
        return None
    return matches.iloc[0]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def compose_model_text(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "passage_text" in data.columns:
        passage = data["passage_text"].where(data["passage_text"].notna(), data["text"])
    else:
        passage = data["text"]
    if "question_text" in data.columns:
        question = data["question_text"].fillna("")
        data["text"] = [
            f"{clean_text(p)}\n\nQuestion: {clean_text(q)}" if clean_text(q) else clean_text(p)
            for p, q in zip(passage, question)
        ]
    else:
        data["text"] = passage.map(clean_text)
    return data


def build_text_features(data: pd.DataFrame, question_col: str) -> pd.DataFrame:
    subset = data.copy()
    subset["reader_id"] = subset["uniform_id"].astype(str)
    subset["trial_id"] = subset["trialid"].astype(int)
    subset["word_str"] = subset["word"].astype(str)
    subset["word_len"] = subset["word_str"].str.len()
    subset["is_punct"] = subset["word_str"].str.fullmatch(r"\W+").fillna(False).astype(float)
    if question_col:
        subset["question_id"] = normalize_question_ids(subset[question_col])
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
