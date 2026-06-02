#!/usr/bin/env python
"""Download the shared MECO data folder from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1zMmdLosiGM8ZY2LNu22yKS8ZyVZM6Kj5?usp=sharing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEFAULT_DRIVE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--remaining-ok", action="store_true")
    parser.add_argument("--no-unzip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        args.url,
        "-O",
        str(args.output_dir),
    ]
    if args.remaining_ok and gdown_supports_remaining_ok():
        command.append("--remaining-ok")
    elif args.remaining_ok:
        print("Installed gdown does not support --remaining-ok; continuing without it.", flush=True)
    subprocess.run(command, check=True)
    if not args.no_unzip:
        for zip_path in args.output_dir.rglob("*.zip"):
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


if __name__ == "__main__":
    main()
