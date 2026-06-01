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

REQUIRED_COLUMNS = [
    "uniform_id",
    "trialid",
    "sentnum",
    "wordnum",
    "word",
    "dur",
    "lang",
]


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
    return data.sort_values(["uniform_id", "trialid", "sentnum", "wordnum"]).reset_index(drop=True)


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
) -> pd.DataFrame:
    feature_names = feature_names or [col for col in BEHAVIOR_PROFILE_COLUMNS if col in data.columns]
    if not feature_names:
        raise ValueError("No behavior profile columns are available")
    profile_trials = set(parse_int_list(profile_trials))
    profile_data = data.loc[data["trialid"].isin(profile_trials)].copy()
    if profile_data.empty:
        raise ValueError(f"No rows found for profile trials: {sorted(profile_trials)}")
    profiles = profile_data.groupby("uniform_id")[feature_names].mean()
    profiles = profiles.reindex(sorted(data["uniform_id"].unique()))
    profiles = profiles.fillna(profiles.mean()).fillna(0.0)
    return profiles.astype(float)


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
        words = sentence.sort_values("wordnum")["word"].astype(str).tolist()
        labels = _make_log_trt_labels(sentence.sort_values("wordnum")["dur"])
        examples.append(
            {
                "tokens": words,
                "labels": labels,
                "reader_profile": profile_lookup[str(reader)].astype(float).tolist(),
                "reader_id": str(reader),
                "trial_id": int(trial_id),
                "sent_num": int(sent_num),
            }
        )
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
        return tokenized

    remove_columns = [col for col in dataset.column_names if col not in {"reader_profile"}]
    return dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)


def build_part1_raw_datasets(
    data: pd.DataFrame,
    profile_trials: Iterable[int],
    train_trials: Iterable[int],
    dev_trials: Iterable[int],
    test_reader_frac: float,
    seed: int,
    explicit_test_readers: Iterable[str] | None = None,
) -> Part1RawDatasets:
    split = make_reader_split(
        data["uniform_id"].unique(),
        test_reader_frac=test_reader_frac,
        seed=seed,
        explicit_test_readers=explicit_test_readers,
    )
    raw_profiles = compute_behavior_profiles(data, profile_trials=profile_trials)
    profile_stats = fit_profile_stats(raw_profiles, split.train_readers)
    profiles = normalize_profiles(raw_profiles, profile_stats)

    train_examples = make_sentence_examples(
        data,
        readers=split.train_readers,
        trials=train_trials,
        profiles=profiles,
        profile_mode="actual",
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
                seed=seed,
            )
        )
    return Part1RawDatasets(train=train, dev=dev, test=test, split=split, profile_stats=profile_stats)


def _make_log_trt_labels(duration_ms: pd.Series) -> list[float | None]:
    labels: list[float | None] = []
    for value in pd.to_numeric(duration_ms, errors="coerce"):
        labels.append(None if pd.isna(value) else float(np.log1p(value)))
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

