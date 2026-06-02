#!/usr/bin/env python
"""Download the VAD dataset archive from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VAD_DRIVE_URL = "https://drive.google.com/uc?id=1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=DEFAULT_VAD_DRIVE_URL)
    parser.add_argument("--output-path", type=Path, default=Path("data") / "auxiliary files" / "Archive (1).zip")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        print(f"VAD archive already exists: {args.output_path}")
        return
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown", args.url, "-O", str(args.output_path)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
