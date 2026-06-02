"""Data preparation for behavior-conditioned MECO TRT prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyreadr
from datasets import Dataset


BEHAVIOR_PROFILE_COLUMNS = [
    "skip",
    "firstrun.skip",
    "reread",
    "refix",
    "reg.in",
    "reg.out",
    "firstrun.refix",
    "firstrun.reg.in",
    "firstrun.reg.out",
]

FULL_GAZE_PROFILE_COLUMNS = [
    "blink",
    "skip",
    "nrun",
    "reread",
    "nfix",
    "refix",
    "reg.in",
    "reg.out",
    "dur",
    "firstrun.skip",
    "firstrun.nfix",
    "firstrun.refix",
    "firstrun.reg.in",
    "firstrun.reg.out",
    "firstrun.dur",
    "firstrun.gopast",
    "firstrun.gopast.sel",
    "firstfix.launch",
    "firstfix.land",
    "firstfix.cland",
    "firstfix.dur",
    "singlefix",
    "singlefix.launch",
    "singlefix.land",
    "singlefix.cland",
    "singlefix.dur",
]

VAD_WORD_TOKEN_FEATURE_COLUMNS = [
    "vad_word_valence",
    "vad_word_arousal",
    "vad_word_dominance",
    "vad_word_valence_centered",
    "vad_word_arousal_centered",
    "vad_word_dominance_centered",
    "vad_word_affective_extremity",
    "vad_word_affective_intensity",
    "vad_word_covered",
]

VAD_SENTENCE_TOKEN_FEATURE_COLUMNS = [
    "vad_sent_valence_mean",
    "vad_sent_arousal_mean",
    "vad_sent_dominance_mean",
    "vad_sent_valence_centered_mean",
    "vad_sent_arousal_centered_mean",
    "vad_sent_dominance_centered_mean",
    "vad_sent_affective_extremity_mean",
    "vad_sent_affective_intensity_mean",
    "vad_sent_arousal_max",
    "vad_sent_valence_min",
    "vad_sent_coverage_rate",
]

VAD_SENSITIVITY_OUTCOMES = {
    "log_trt": "log_trt",
    "nfix": "nfix",
    "firstrun_log_dur": "firstrun_log_dur",
    "reread": "reread",
}

VAD_SENSITIVITY_PREDICTORS = [
    "vad_word_valence_centered",
    "vad_word_arousal_centered",
    "vad_word_dominance_centered",
    "vad_word_affective_intensity_centered",
]

REQUIRED_COLUMNS = [
    "uniform_id",
    "trialid",
    "sentnum",
    "wordnum",
    "word",
    "dur",
    "lang",
]

BASELINE_KEY_COLUMNS = ["uniform_id", "trialid", "sentnum", "wordnum"]


def token_feature_columns_for_set(token_feature_set: str) -> list[str]:
    if token_feature_set == "none":
        return []
    if token_feature_set == "vad_word":
        return list(VAD_WORD_TOKEN_FEATURE_COLUMNS)
    if token_feature_set == "vad_word_sentence":
        return [*VAD_WORD_TOKEN_FEATURE_COLUMNS, *VAD_SENTENCE_TOKEN_FEATURE_COLUMNS]
    raise ValueError(f"Unknown token_feature_set: {token_feature_set}")


@dataclass(frozen=True)
class ReaderSplit:
    train_readers: list[str]
    test_readers: list[str]


@dataclass(frozen=True)
class ProfileStats:
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class Part1RawDatasets:
    train: Dataset
    dev: dict[str, Dataset]
    test: dict[str, Dataset]
    split: ReaderSplit
    profile_stats: ProfileStats


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def load_meco_rda(path: str | Path, lang: str = "en") -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No data frames found in {path}")
    data = next(iter(result.values()))
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"MECO data is missing required columns: {sorted(missing)}")
    data = data.loc[data["lang"].astype(str) == lang].copy()
    if data.empty:
        raise ValueError(f"No rows found for lang={lang!r}")
    data["uniform_id"] = data["uniform_id"].astype(str)
    data["trialid"] = data["trialid"].astype(int)
    data["sentnum"] = data["sentnum"].astype(int)
    data["wordnum"] = data["wordnum"].astype(int)
    data["word"] = data["word"].astype(str)
    data["dur"] = pd.to_numeric(data["dur"], errors="coerce")
    for col in BEHAVIOR_PROFILE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    for col in FULL_GAZE_PROFILE_COLUMNS:
        if col in data.columns and col not in BEHAVIOR_PROFILE_COLUMNS and col != "dur":
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values(["uniform_id", "trialid", "sentnum", "wordnum"]).reset_index(drop=True)


def load_vad_features(path: str | Path) -> pd.DataFrame:
    vad = pd.read_csv(path)
    missing = set(BASELINE_KEY_COLUMNS).difference(vad.columns)
    if missing:
        raise ValueError(f"VAD features are missing required key columns: {sorted(missing)}")
    vad = vad.copy()
    vad["uniform_id"] = vad["uniform_id"].astype(str)
    for col in ["trialid", "sentnum", "wordnum"]:
        vad[col] = pd.to_numeric(vad[col], errors="coerce").astype("Int64")
    vad = vad.dropna(subset=BASELINE_KEY_COLUMNS).copy()
    for col in ["trialid", "sentnum", "wordnum"]:
        vad[col] = vad[col].astype(int)

    duplicated = vad.duplicated(BASELINE_KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = vad.loc[duplicated, BASELINE_KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"VAD feature file contains duplicated MECO keys. Examples: {examples}")

    for col in vad.columns:
        if col not in BASELINE_KEY_COLUMNS and col not in {"word", "word_norm"}:
            vad[col] = pd.to_numeric(vad[col], errors="coerce")
    return vad.sort_values(BASELINE_KEY_COLUMNS).reset_index(drop=True)


def attach_vad_features(data: pd.DataFrame, vad_features: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    vad_features = vad_features.drop(columns=["word", "word_norm"], errors="ignore")
    overlap = [
        col
        for col in vad_features.columns
        if col in data.columns and col not in BASELINE_KEY_COLUMNS and col not in {"word", "word_norm"}
    ]
    if overlap:
        data = data.drop(columns=overlap)
    return data.merge(vad_features, on=BASELINE_KEY_COLUMNS, how="left", validate="one_to_one")


def ensure_log_trt_column(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["log_trt"] = [
        None if pd.isna(value) else float(np.log1p(value))
        for value in pd.to_numeric(data["dur"], errors="coerce")
    ]
    return data


def load_residual_baseline_predictions(path: str | Path) -> pd.DataFrame:
    predictions_dir = Path(path)
    if (predictions_dir / "predictions").exists():
        predictions_dir = predictions_dir / "predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Residual baseline predictions directory not found: {predictions_dir}")

    all_files = sorted(predictions_dir.glob("*_predictions.csv"))
    actual_files = [file for file in all_files if file.name.endswith("_actual_predictions.csv")]
    selected_files = actual_files if actual_files else all_files
    if not selected_files:
        raise FileNotFoundError(f"No baseline prediction CSV files found under {predictions_dir}")

    frames = []
    required = {"reader_id", "trial_id", "sent_num", "word_num", "true_log_trt", "pred_log_trt"}
    for file in selected_files:
        frame = pd.read_csv(file)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Baseline prediction file {file} is missing columns: {sorted(missing)}")
        frame = frame.rename(
            columns={
                "reader_id": "uniform_id",
                "trial_id": "trialid",
                "sent_num": "sentnum",
                "word_num": "wordnum",
                "true_log_trt": "baseline_true_log_trt",
                "pred_log_trt": "baseline_pred_log_trt",
            }
        )
        frames.append(frame[[*BASELINE_KEY_COLUMNS, "baseline_true_log_trt", "baseline_pred_log_trt"]])

    baseline = pd.concat(frames, ignore_index=True)
    baseline["uniform_id"] = baseline["uniform_id"].astype(str)
    for col in ["trialid", "sentnum", "wordnum"]:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce").astype("Int64")
    baseline["baseline_true_log_trt"] = pd.to_numeric(baseline["baseline_true_log_trt"], errors="coerce")
    baseline["baseline_pred_log_trt"] = pd.to_numeric(baseline["baseline_pred_log_trt"], errors="coerce")
    baseline = baseline.dropna(subset=BASELINE_KEY_COLUMNS + ["baseline_pred_log_trt"]).copy()
    for col in ["trialid", "sentnum", "wordnum"]:
        baseline[col] = baseline[col].astype(int)

    duplicated = baseline.duplicated(BASELINE_KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = baseline.loc[duplicated, BASELINE_KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(
            "Residual baseline predictions contain duplicated "
            f"(reader_id, trial_id, sent_num, word_num) rows. Examples: {examples}"
        )
    return baseline.sort_values(BASELINE_KEY_COLUMNS).reset_index(drop=True)


def attach_residual_baseline(data: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    data = ensure_log_trt_column(data)
    merged = data.merge(baseline, on=BASELINE_KEY_COLUMNS, how="left", validate="many_to_one")
    merged["residual_log_trt"] = merged["log_trt"] - merged["baseline_pred_log_trt"]
    return merged


def make_reader_split(
    readers: Iterable[str],
    test_reader_frac: float,
    seed: int,
    explicit_test_readers: Iterable[str] | None = None,
) -> ReaderSplit:
    readers = sorted(str(reader) for reader in readers)
    if explicit_test_readers is not None:
        test_readers = sorted(str(reader) for reader in explicit_test_readers)
        unknown = set(test_readers).difference(readers)
        if unknown:
            raise ValueError(f"Unknown test readers: {sorted(unknown)}")
    else:
        rng = np.random.default_rng(seed)
        shuffled = np.array(readers, dtype=object)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(readers) * test_reader_frac)))
        test_readers = sorted(shuffled[:n_test].tolist())
    train_readers = [reader for reader in readers if reader not in set(test_readers)]
    if not train_readers:
        raise ValueError("Reader split left no training readers")
    return ReaderSplit(train_readers=train_readers, test_readers=test_readers)


def compute_behavior_profiles(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
    feature_names: list[str] | None = None,
    profile_feature_set: str = "behavior_only",
) -> pd.DataFrame:
    if feature_names is not None:
        return _compute_mean_profiles(data, profile_trials=profile_trials, feature_names=feature_names)
    if profile_feature_set == "behavior_only":
        feature_names = [col for col in BEHAVIOR_PROFILE_COLUMNS if col in data.columns]
        return _compute_mean_profiles(data, profile_trials=profile_trials, feature_names=feature_names)
    if profile_feature_set == "full_gaze":
        feature_names = [col for col in FULL_GAZE_PROFILE_COLUMNS if col in data.columns]
        return _compute_summary_profiles(data, profile_trials=profile_trials, feature_names=feature_names)
    if profile_feature_set == "full_gaze_vad":
        feature_names = [col for col in FULL_GAZE_PROFILE_COLUMNS if col in data.columns]
        profiles = _compute_summary_profiles(data, profile_trials=profile_trials, feature_names=feature_names)
        vad_profiles = compute_vad_sensitivity_profiles(data, profile_trials=profile_trials)
        return profiles.join(vad_profiles, how="left").fillna(0.0)
    raise ValueError(f"Unknown profile_feature_set: {profile_feature_set}")


def _compute_mean_profiles(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
    feature_names: list[str],
) -> pd.DataFrame:
    if not feature_names:
        raise ValueError("No profile columns are available")
    profile_trials = set(parse_int_list(profile_trials))
    profile_data = data.loc[data["trialid"].isin(profile_trials)].copy()
    if profile_data.empty:
        raise ValueError(f"No rows found for profile trials: {sorted(profile_trials)}")
    for col in feature_names:
        profile_data[col] = pd.to_numeric(profile_data[col], errors="coerce")
    profiles = profile_data.groupby("uniform_id")[feature_names].mean()
    profiles = profiles.reindex(sorted(data["uniform_id"].unique()))
    profiles = profiles.fillna(profiles.mean()).fillna(0.0)
    return profiles.astype(float)


def _compute_summary_profiles(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
    feature_names: list[str],
) -> pd.DataFrame:
    if not feature_names:
        raise ValueError("No full gaze profile columns are available")
    profile_trials = set(parse_int_list(profile_trials))
    profile_data = data.loc[data["trialid"].isin(profile_trials)].copy()
    if profile_data.empty:
        raise ValueError(f"No rows found for profile trials: {sorted(profile_trials)}")
    for col in feature_names:
        profile_data[col] = pd.to_numeric(profile_data[col], errors="coerce")
    grouped = profile_data.groupby("uniform_id")[feature_names]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().fillna(0.0).add_suffix("_std")
    profiles = pd.concat([mean, std], axis=1)
    profiles = profiles.reindex(sorted(data["uniform_id"].unique()))
    profiles = profiles.fillna(profiles.mean()).fillna(0.0)
    return profiles.astype(float)


def compute_vad_sensitivity_profiles(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
) -> pd.DataFrame:
    missing = [col for col in VAD_SENSITIVITY_PREDICTORS if col not in data.columns]
    if missing:
        raise ValueError(f"full_gaze_vad requires VAD predictor columns: {missing}")
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
    profile_trials = set(parse_int_list(profile_trials))
    profile_data = data.loc[data["trialid"].isin(profile_trials)].copy()
    rows = []
    for reader, group in profile_data.groupby("uniform_id", sort=True):
        row = {"uniform_id": str(reader)}
        for outcome_name, outcome_col in VAD_SENSITIVITY_OUTCOMES.items():
            slopes = _fit_reader_vad_slopes(group, outcome_col=outcome_col)
            for predictor, slope in slopes.items():
                clean_predictor = predictor.replace("vad_word_", "").replace("_centered", "")
                row[f"vad_sensitivity_{outcome_name}_{clean_predictor}"] = float(slope)
        rows.append(row)
    profiles = pd.DataFrame(rows).set_index("uniform_id") if rows else pd.DataFrame()
    profiles = profiles.reindex(sorted(data["uniform_id"].unique()))
    profiles = profiles.fillna(profiles.mean()).fillna(0.0)
    return profiles.astype(float)


def _fit_reader_vad_slopes(group: pd.DataFrame, outcome_col: str) -> dict[str, float]:
    needed = [outcome_col, *VAD_SENSITIVITY_PREDICTORS]
    subset = group[needed].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < len(VAD_SENSITIVITY_PREDICTORS) + 2:
        return {predictor: 0.0 for predictor in VAD_SENSITIVITY_PREDICTORS}
    x = subset[VAD_SENSITIVITY_PREDICTORS].to_numpy(dtype=float)
    y = subset[outcome_col].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {predictor: 0.0 for predictor in VAD_SENSITIVITY_PREDICTORS}
    return {predictor: float(value) for predictor, value in zip(VAD_SENSITIVITY_PREDICTORS, beta[1:])}


def fit_profile_stats(profiles: pd.DataFrame, train_readers: Iterable[str]) -> ProfileStats:
    train_matrix = profiles.loc[list(train_readers)].to_numpy(dtype=np.float32)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ProfileStats(mean=mean.astype(np.float32), std=std.astype(np.float32), feature_names=list(profiles.columns))


def normalize_profiles(profiles: pd.DataFrame, stats: ProfileStats) -> dict[str, np.ndarray]:
    matrix = profiles[stats.feature_names].to_numpy(dtype=np.float32)
    normalized = (matrix - stats.mean) / stats.std
    return {reader: normalized[idx].astype(np.float32) for idx, reader in enumerate(profiles.index)}


def make_sentence_examples(
    data: pd.DataFrame,
    readers: Iterable[str],
    trials: Iterable[int],
    profiles: dict[str, np.ndarray],
    profile_mode: str,
    label_column: str = "log_trt",
    token_feature_columns: list[str] | None = None,
    seed: int = 13,
) -> list[dict[str, Any]]:
    readers = sorted(str(reader) for reader in readers)
    trials = set(parse_int_list(trials))
    subset = data.loc[data["uniform_id"].isin(readers) & data["trialid"].isin(trials)].copy()
    if subset.empty:
        raise ValueError(f"No rows for readers={len(readers)} and trials={sorted(trials)}")

    profile_lookup = _build_profile_lookup(readers, profiles, profile_mode=profile_mode, seed=seed)
    examples: list[dict[str, Any]] = []
    group_cols = ["uniform_id", "trialid", "sentnum"]
    for (reader, trial_id, sent_num), sentence in subset.groupby(group_cols, sort=True):
        sentence = sentence.sort_values("wordnum")
        words = sentence["word"].astype(str).tolist()
        word_nums = sentence["wordnum"].astype(int).tolist()
        if label_column not in sentence.columns:
            if label_column == "log_trt":
                labels = _make_log_trt_labels(sentence["dur"])
            else:
                raise ValueError(f"Missing label column: {label_column}")
        else:
            labels = _make_numeric_labels(sentence[label_column])
        example = {
            "tokens": words,
            "word_nums": word_nums,
            "labels": labels,
            "reader_profile": profile_lookup[str(reader)].astype(float).tolist(),
            "reader_id": str(reader),
            "trial_id": int(trial_id),
            "sent_num": int(sent_num),
        }
        if "log_trt" in sentence.columns:
            example["raw_log_trt"] = _make_numeric_labels(sentence["log_trt"])
        if "baseline_pred_log_trt" in sentence.columns:
            example["baseline_log_trt"] = _make_numeric_labels(sentence["baseline_pred_log_trt"])
        if token_feature_columns:
            missing = [col for col in token_feature_columns if col not in sentence.columns]
            if missing:
                raise ValueError(f"Missing token feature columns: {missing}")
            token_features = (
                sentence[token_feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            example["token_features"] = token_features.astype(float).tolist()
        examples.append(example)
    return examples


def tokenize_and_align_examples(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        aligned_labels = []
        for batch_idx, word_labels in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_idx)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    label_ids.append(-100.0)
                else:
                    value = word_labels[word_idx]
                    label_ids.append(-100.0 if value is None or not np.isfinite(value) else float(value))
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        tokenized["reader_profile"] = batch["reader_profile"]
        if "token_features" in batch:
            aligned_features = []
            feature_dim = len(batch["token_features"][0][0]) if batch["token_features"] and batch["token_features"][0] else 0
            zero_features = [0.0] * feature_dim
            for batch_idx, word_features in enumerate(batch["token_features"]):
                word_ids = tokenized.word_ids(batch_index=batch_idx)
                previous_word_idx = None
                feature_ids = []
                for word_idx in word_ids:
                    if word_idx is None or word_idx == previous_word_idx:
                        feature_ids.append(zero_features)
                    else:
                        feature_ids.append(word_features[word_idx])
                    previous_word_idx = word_idx
                aligned_features.append(feature_ids)
            tokenized["token_features"] = aligned_features
        return tokenized

    remove_columns = [col for col in dataset.column_names if col not in {"reader_profile", "token_features"}]
    return dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)


def build_part1_raw_datasets(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
    train_trials: Iterable[int],
    dev_trials: Iterable[int],
    test_reader_frac: float,
    seed: int,
    explicit_test_readers: Iterable[str] | None = None,
    label_column: str = "log_trt",
    profile_feature_set: str = "behavior_only",
    token_feature_columns: list[str] | None = None,
) -> Part1RawDatasets:
    if label_column == "log_trt" and "log_trt" not in data.columns:
        data = ensure_log_trt_column(data)
    split = make_reader_split(
        data["uniform_id"].unique(),
        test_reader_frac=test_reader_frac,
        seed=seed,
        explicit_test_readers=explicit_test_readers,
    )
    raw_profiles = compute_behavior_profiles(data, profile_trials=profile_trials, profile_feature_set=profile_feature_set)
    profile_stats = fit_profile_stats(raw_profiles, split.train_readers)
    profiles = normalize_profiles(raw_profiles, profile_stats)

    train_examples = make_sentence_examples(
        data,
        readers=split.train_readers,
        trials=train_trials,
        profiles=profiles,
        profile_mode="actual",
        label_column=label_column,
        token_feature_columns=token_feature_columns,
        seed=seed,
    )
    train = Dataset.from_list(train_examples)

    dev: dict[str, Dataset] = {}
    test: dict[str, Dataset] = {}
    for mode in ["actual", "mean", "shuffled"]:
        dev[mode] = Dataset.from_list(
            make_sentence_examples(
                data,
                readers=split.train_readers,
                trials=dev_trials,
                profiles=profiles,
                profile_mode=mode,
                label_column=label_column,
                token_feature_columns=token_feature_columns,
                seed=seed,
            )
        )
        test[mode] = Dataset.from_list(
            make_sentence_examples(
                data,
                readers=split.test_readers,
                trials=dev_trials,
                profiles=profiles,
                profile_mode=mode,
                label_column=label_column,
                token_feature_columns=token_feature_columns,
                seed=seed,
            )
        )
    return Part1RawDatasets(train=train, dev=dev, test=test, split=split, profile_stats=profile_stats)


def _make_log_trt_labels(duration_ms: pd.Series) -> list[float | None]:
    return _make_numeric_labels(pd.Series(np.log1p(pd.to_numeric(duration_ms, errors="coerce"))))


def _make_numeric_labels(values: pd.Series) -> list[float | None]:
    labels: list[float | None] = []
    for value in pd.to_numeric(values, errors="coerce"):
        labels.append(None if pd.isna(value) or not np.isfinite(value) else float(value))
    return labels


def _build_profile_lookup(
    readers: list[str],
    profiles: dict[str, np.ndarray],
    profile_mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    profile_dim = len(next(iter(profiles.values())))
    if profile_mode == "actual":
        return {reader: profiles[reader] for reader in readers}
    if profile_mode == "mean":
        mean_profile = np.zeros(profile_dim, dtype=np.float32)
        return {reader: mean_profile for reader in readers}
    if profile_mode == "shuffled":
        rng = np.random.default_rng(seed)
        shuffled = list(readers)
        rng.shuffle(shuffled)
        if len(shuffled) > 1 and any(a == b for a, b in zip(readers, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
        return {reader: profiles[other] for reader, other in zip(readers, shuffled)}
    raise ValueError(f"Unknown profile_mode: {profile_mode}")
