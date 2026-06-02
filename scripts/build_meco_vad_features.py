#!/usr/bin/env python
"""Build word- and sentence-level VAD features for English MECO rows."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from part1_behavior_conditioned.data import BASELINE_KEY_COLUMNS, load_meco_rda


TARGET_RDA_NAME = "joint_l1_data_trimmed_version1.3.rda"
DEFAULT_SOURCES = ["nrc_vad.tsv", "warriner_et_al.tsv", "scott_et_al.tsv"]
DEFAULT_VAD_DRIVE_URL = "https://drive.google.com/file/d/1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj/view?usp=sharing"
DEFAULT_VAD_ARCHIVE_PATH = Path("data") / "auxiliary files" / "Archive.zip"
VAD_ARCHIVE_FALLBACK_NAMES = ["Archive.zip", "Archive (1).zip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rda-path", type=Path, default=Path("data") / "primary data" / "eye tracking data" / TARGET_RDA_NAME)
    parser.add_argument("--vad-archive-path", type=Path, default=DEFAULT_VAD_ARCHIVE_PATH)
    parser.add_argument("--vad-drive-url", type=str, default=DEFAULT_VAD_DRIVE_URL)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/vad/meco_en_vad_features.csv"))
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rda_path = resolve_rda_path(args.rda_path)
    vad_archive_path = ensure_vad_archive(args.vad_archive_path, args.vad_drive_url, allow_download=not args.no_download)
    meco = load_meco_rda(rda_path, lang=args.lang)
    lexicon, source_report = load_combined_lexicon(vad_archive_path, args.sources)
    features = build_features(meco, lexicon)
    validate_features(features)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_path, index=False)
    manifest = {
        "rda_path": str(rda_path),
        "vad_archive_path": str(vad_archive_path),
        "output_path": str(args.output_path),
        "language": args.lang,
        "sources": args.sources,
        "source_report": source_report,
        "n_rows": int(len(features)),
        "n_unique_words": int(features["word_norm"].nunique()),
        "word_coverage_rate": float(features["vad_word_covered"].mean()),
        "sentence_mean_coverage_rate": float(features["vad_sent_coverage_rate"].mean()),
    }
    manifest_path = args.output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def ensure_vad_archive(path: Path, url: str, allow_download: bool) -> Path:
    for candidate in archive_candidates(path):
        if candidate.exists():
            if candidate != path:
                print(f"Using existing VAD archive: {candidate}", flush=True)
            return candidate
    if not allow_download:
        raise FileNotFoundError(f"VAD archive not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown"]
    add_gdown_flag(command, "--fuzzy")
    add_gdown_flag(command, "--no-cookies")
    add_gdown_flag(command, "--continue")
    command.extend([url, "-O", str(path)])
    print("Downloading VAD archive:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not path.exists():
        raise FileNotFoundError(f"gdown finished but VAD archive was not created: {path}")
    return path


def archive_candidates(path: Path) -> list[Path]:
    candidates = [path]
    for name in VAD_ARCHIVE_FALLBACK_NAMES:
        candidate = path.parent / name
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def gdown_supports_flag(flag: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "gdown", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return flag in result.stdout


def add_gdown_flag(command: list[str], flag: str) -> None:
    if gdown_supports_flag(flag):
        command.append(flag)


def resolve_rda_path(path: Path) -> Path:
    if path.exists():
        return path
    candidates = sorted(Path("data").rglob(TARGET_RDA_NAME))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Could not find {path}. Try: find data -name '{TARGET_RDA_NAME}' -print")


def load_combined_lexicon(path: Path, sources: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if not path.exists():
        raise FileNotFoundError(f"VAD archive not found: {path}")
    frames = []
    report = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for source in sources:
            if source not in names:
                raise FileNotFoundError(f"{source} not found inside {path}")
            with archive.open(source) as handle:
                frame = pd.read_csv(handle, sep="\t")
            required = {"text", "valence", "arousal", "dominance"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{source} is missing columns: {sorted(missing)}")
            frame = frame[["text", "valence", "arousal", "dominance"]].copy()
            frame["source"] = source.replace(".tsv", "")
            frame["word_norm"] = frame["text"].map(normalize_word)
            frame = frame.loc[frame["word_norm"] != ""].copy()
            for col in ["valence", "arousal", "dominance"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").clip(0.0, 1.0)
            frame = frame.dropna(subset=["valence", "arousal", "dominance"])
            frame = frame.groupby(["source", "word_norm"], as_index=False).agg(
                valence=("valence", "mean"),
                arousal=("arousal", "mean"),
                dominance=("dominance", "mean"),
            )
            report[source] = {"n_rows": int(len(frame)), "n_words": int(frame["word_norm"].nunique())}
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    lexicon = combined.groupby("word_norm", as_index=False).agg(
        vad_word_valence=("valence", "mean"),
        vad_word_arousal=("arousal", "mean"),
        vad_word_dominance=("dominance", "mean"),
        vad_word_source_count=("source", "nunique"),
    )
    return lexicon, report


def build_features(meco: pd.DataFrame, lexicon: pd.DataFrame) -> pd.DataFrame:
    features = meco[[*BASELINE_KEY_COLUMNS, "word"]].copy()
    features["word_norm"] = features["word"].map(normalize_word)
    features = features.merge(lexicon, on="word_norm", how="left", validate="many_to_one")
    features["vad_word_source_count"] = features["vad_word_source_count"].fillna(0).astype(int)
    features["vad_word_covered"] = (features["vad_word_source_count"] > 0).astype(float)
    features["vad_word_valence_centered"] = features["vad_word_valence"] - 0.5
    features["vad_word_arousal_centered"] = features["vad_word_arousal"] - 0.5
    features["vad_word_dominance_centered"] = features["vad_word_dominance"] - 0.5
    features["vad_word_affective_extremity"] = features["vad_word_valence_centered"].abs()
    features["vad_word_affective_intensity"] = np.sqrt(
        features[
            [
                "vad_word_valence_centered",
                "vad_word_arousal_centered",
                "vad_word_dominance_centered",
            ]
        ]
        .pow(2)
        .sum(axis=1, skipna=False)
    )
    intensity_mean = features.loc[features["vad_word_covered"] > 0, "vad_word_affective_intensity"].mean()
    features["vad_word_affective_intensity_centered"] = features["vad_word_affective_intensity"] - float(intensity_mean)
    sentence_features = build_sentence_features(features)
    return features.merge(sentence_features, on=["uniform_id", "trialid", "sentnum"], how="left", validate="many_to_one")


def build_sentence_features(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in features.groupby(["uniform_id", "trialid", "sentnum"], sort=True):
        covered = group.loc[group["vad_word_covered"] > 0].copy()
        row = {
            "uniform_id": keys[0],
            "trialid": keys[1],
            "sentnum": keys[2],
            "vad_sent_coverage_rate": float(group["vad_word_covered"].mean()),
        }
        if covered.empty:
            row.update(
                {
                    "vad_sent_valence_mean": np.nan,
                    "vad_sent_arousal_mean": np.nan,
                    "vad_sent_dominance_mean": np.nan,
                    "vad_sent_valence_centered_mean": np.nan,
                    "vad_sent_arousal_centered_mean": np.nan,
                    "vad_sent_dominance_centered_mean": np.nan,
                    "vad_sent_affective_extremity_mean": np.nan,
                    "vad_sent_affective_intensity_mean": np.nan,
                    "vad_sent_arousal_max": np.nan,
                    "vad_sent_valence_min": np.nan,
                }
            )
        else:
            row.update(
                {
                    "vad_sent_valence_mean": float(covered["vad_word_valence"].mean()),
                    "vad_sent_arousal_mean": float(covered["vad_word_arousal"].mean()),
                    "vad_sent_dominance_mean": float(covered["vad_word_dominance"].mean()),
                    "vad_sent_valence_centered_mean": float(covered["vad_word_valence_centered"].mean()),
                    "vad_sent_arousal_centered_mean": float(covered["vad_word_arousal_centered"].mean()),
                    "vad_sent_dominance_centered_mean": float(covered["vad_word_dominance_centered"].mean()),
                    "vad_sent_affective_extremity_mean": float(covered["vad_word_affective_extremity"].mean()),
                    "vad_sent_affective_intensity_mean": float(covered["vad_word_affective_intensity"].mean()),
                    "vad_sent_arousal_max": float(covered["vad_word_arousal"].max()),
                    "vad_sent_valence_min": float(covered["vad_word_valence"].min()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_features(features: pd.DataFrame) -> None:
    duplicated = features.duplicated(BASELINE_KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = features.loc[duplicated, BASELINE_KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"VAD features contain duplicated MECO keys. Examples: {examples}")
    if not features["vad_word_covered"].isin([0.0, 1.0]).all():
        raise ValueError("vad_word_covered must be binary")


def normalize_word(value: object) -> str:
    text = str(value).lower().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"^[^\w']+|[^\w']+$", "", text)
    return text


if __name__ == "__main__":
    main()
