"""Build the sheet that decides Gate 1.

    python -m avatar.cli.likeness blind --photos DIR --renders DIR -o sheet.png
    python -m avatar.cli.likeness comparison --photos DIR --renders DIR -o sheet.png

Use `blind` for the real test. Show it to somebody who knew the person, without
telling them what they are looking at, and ask whether every tile is the same
person and which are the odd ones out. `comparison` is for looking at your own
work, not for asking anybody anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from avatar.likeness.sheet import blind, comparison

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")
    found = sorted(p for p in directory.iterdir() if p.suffix.lower() in SUFFIXES)
    if not found:
        raise SystemExit(f"no images in {directory}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["blind", "comparison"])
    parser.add_argument("--photos", required=True, type=Path)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("-o", "--out", default=Path("likeness-sheet.png"), type=Path)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    photographs = images_in(args.photos)
    renders = images_in(args.renders)

    sheet = (
        blind(photographs, renders, seed=args.seed)
        if args.mode == "blind"
        else comparison(photographs, renders)
    )
    path = sheet.save(args.out)

    print(f"\n{path}")
    if args.mode == "blind":
        print(f"answers: {path.with_suffix('.key.json')}")
        print("\nAsk: are these all the same person? Which are the odd ones out?")
        print("Do not say which tiles are rendered.")


if __name__ == "__main__":
    sys.exit(main())
