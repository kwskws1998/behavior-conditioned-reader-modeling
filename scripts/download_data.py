#!/usr/bin/env python
"""Download the shared MECO data folder from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1zMmdLosiGM8ZY2LNu22yKS8ZyVZM6Kj5?usp=sharing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEFAULT_DRIVE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--remaining-ok", action="store_true")
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
    if args.remaining_ok:
        command.append("--remaining-ok")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

