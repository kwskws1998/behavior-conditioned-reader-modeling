#!/usr/bin/env python
"""Download the shared MECO data folder and VAD archive from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1zMmdLosiGM8ZY2LNu22yKS8ZyVZM6Kj5?usp=sharing"
DEFAULT_VAD_DRIVE_URL = "https://drive.google.com/file/d/1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj/view?usp=sharing"
DEFAULT_VAD_OUTPUT_PATH = Path("data") / "auxiliary files" / "Archive.zip"
VAD_ARCHIVE_FALLBACK_NAMES = ["Archive.zip", "Archive (1).zip"]
ESSENTIAL_MECO_FILES = [
    (
        "1OeYJ1JTSGoZVJJpeafFCMRPuBGHrHHjy",
        Path("primary data") / "eye tracking data" / "joint_l1_data_trimmed_version1.3.rda",
    ),
    (
        "12Ae_InyNwnEzuaLyTooRD73aw_ewnO4d",
        Path("primary data") / "comprehension data" / "joint_l1_acc_full_breakdown.rda",
    ),
    (
        "11B4Mw4TIKPhuIkiSsXb6vGtK9_y7zO7M",
        Path("auxiliary files") / "reading task materials" / "comp-questions.xlsx",
    ),
    (
        "1z7YykDgpsqO_ygIZndjona32fL_7I5V9",
        Path("auxiliary files") / "reading task materials" / "supp texts.xlsx",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEFAULT_DRIVE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--vad-url", type=str, default=DEFAULT_VAD_DRIVE_URL)
    parser.add_argument("--vad-output-path", type=Path, default=DEFAULT_VAD_OUTPUT_PATH)
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--essential-only", action="store_true")
    parser.add_argument("--remaining-ok", action="store_true")
    parser.add_argument("--no-unzip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.essential_only:
        download_essential_meco_files(args.output_dir)
    else:
        command = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            args.url,
            "-O",
            str(args.output_dir),
        ]
        add_gdown_flag(command, "--no-cookies")
        add_gdown_flag(command, "--continue")
        if args.remaining_ok and gdown_supports_remaining_ok():
            command.append("--remaining-ok")
        elif args.remaining_ok:
            print("Installed gdown does not support --remaining-ok; continuing without it.", flush=True)
        try:
            run_gdown(command)
        except subprocess.CalledProcessError:
            print("Folder download failed; falling back to essential MECO files.", flush=True)
            download_essential_meco_files(args.output_dir)

    vad_archive_path = None
    if not args.skip_vad:
        vad_archive_path = ensure_vad_archive(args.vad_output_path, args.vad_url, args.output_dir)

    if not args.no_unzip:
        for zip_path in args.output_dir.rglob("*.zip"):
            if vad_archive_path is not None and zip_path.resolve() == vad_archive_path.resolve():
                continue
            print(f"Extracting {zip_path} into {args.output_dir}", flush=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(args.output_dir)


def gdown_supports_remaining_ok() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "gdown", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "--remaining-ok" in result.stdout


def ensure_vad_archive(path: Path, url: str, output_dir: Path) -> Path:
    for candidate in vad_archive_candidates(path, output_dir):
        if candidate.exists():
            if candidate != path:
                print(f"Using existing VAD archive: {candidate}", flush=True)
            return candidate

    path.parent.mkdir(parents=True, exist_ok=True)
    download_single_file(url, path)
    if not path.exists():
        raise FileNotFoundError(f"gdown finished but VAD archive was not created: {path}")
    return path


def download_essential_meco_files(output_dir: Path) -> None:
    for file_id, relative_path in ESSENTIAL_MECO_FILES:
        destination = output_dir / relative_path
        if destination.exists():
            print(f"Already exists, skip download: {destination}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        download_single_file(url, destination)


def download_single_file(url: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError:
        command = [sys.executable, "-m", "gdown"]
        add_gdown_flag(command, "--no-cookies")
        add_gdown_flag(command, "--continue")
        command.extend([url, "-O", str(destination)])
        run_gdown(command)
        return

    print(f"Downloading file: {url} -> {destination}", flush=True)
    try:
        downloaded = gdown.download(url, str(destination), quiet=False, fuzzy=True)
    except TypeError:
        downloaded = gdown.download(url, str(destination), quiet=False)
    if not downloaded or not destination.exists():
        raise FileNotFoundError(f"gdown finished but file was not created: {destination}")


def vad_archive_candidates(path: Path, output_dir: Path) -> list[Path]:
    candidates = [path]
    for name in VAD_ARCHIVE_FALLBACK_NAMES:
        candidate = path.parent / name
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in sorted(output_dir.rglob("Archive*.zip")):
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


def run_gdown(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if "--no-cookies" in command:
            raise
        retry = command.copy()
        insert_at = retry.index("-O") if "-O" in retry else len(retry)
        retry.insert(insert_at, "--no-cookies")
        print("gdown failed; retrying without cookies:", " ".join(retry), flush=True)
        subprocess.run(retry, check=True)


if __name__ == "__main__":
    main()
