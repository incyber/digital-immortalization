"""The likeness test, made possible to actually run.

Gate 1 of the 3D avatar plan decides whether the product is viable: does a face
built from a family's photographs read as that person to somebody who knew
them. It is decided by a human, not by a metric, and the whole point is to ask
them without tipping them off.

So this builds a contact sheet, and it is deliberately dumb. No scoring, no
similarity number, no highlighting of the render. A quantitative face-identity
score would also drag in a face-recognition model whose weights are almost
always research-licensed - the same trap as InsightFace - to answer a question
a person answers better anyway.

Two modes, for two different questions:

    comparison  photographs on top, renders below, labelled.
                "Here is the person, here is our attempt. Is it him?"

    blind       everything shuffled into one grid, unlabelled, with the answer
                key written separately. "Which of these is the same person?"

Blind is the one that matters. Anyone who watched the render being built cannot
un-know which tile it is, and a labelled sheet only ever collects politeness.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

TILE = 320
COLUMNS = 4
MARGIN = 12
LABEL_HEIGHT = 28

BACKGROUND = (18, 18, 18)
LABEL_COLOUR = (230, 230, 230)


@dataclass(frozen=True)
class Sheet:
    image: np.ndarray
    key: list[dict]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.image)
        # Written beside the sheet rather than into it, so the sheet can be
        # shown to someone without the answers being visible over their
        # shoulder.
        path.with_suffix(".key.json").write_text(json.dumps(self.key, indent=2))
        logger.info(f"sheet {path} with {len(self.key)} tiles")
        return path


def _fit(image: np.ndarray, size: int = TILE) -> np.ndarray:
    """Square, centred, no distortion.

    Letterboxed rather than stretched. A face squeezed by an aspect-ratio
    change is a different face, and this sheet exists to answer whether two
    faces are the same one.
    """
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )

    canvas = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    strip = np.full((LABEL_HEIGHT, tile.shape[1], 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        strip, text, (6, LABEL_HEIGHT - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOUR, 1, cv2.LINE_AA,
    )
    return np.vstack([tile, strip])


def _grid(tiles: list[np.ndarray], columns: int = COLUMNS) -> np.ndarray:
    if not tiles:
        raise ValueError("nothing to lay out")

    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        # The last row is padded so the sheet stays rectangular. A ragged edge
        # draws the eye to the final tile, which on a blind sheet is a hint.
        while len(row) < columns:
            row.append(np.full_like(tiles[0], BACKGROUND))
        rows.append(np.hstack([
            np.hstack([t, np.full((t.shape[0], MARGIN, 3), BACKGROUND, np.uint8)])
            for t in row
        ]))

    width = max(r.shape[1] for r in rows)
    spacer = np.full((MARGIN, width, 3), BACKGROUND, dtype=np.uint8)

    stacked = []
    for row in rows:
        if row.shape[1] < width:
            pad = np.full((row.shape[0], width - row.shape[1], 3), BACKGROUND, np.uint8)
            row = np.hstack([row, pad])
        stacked.extend([row, spacer])
    return np.vstack(stacked)


def _read(paths: list[Path]) -> list[tuple[Path, np.ndarray]]:
    out = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"could not read {path}")
            continue
        out.append((path, image))
    if not out:
        raise ValueError("none of those images could be read")
    return out


def comparison(photographs: list[Path], renders: list[Path]) -> Sheet:
    """Photographs above, renders below, both labelled."""
    tiles, key = [], []

    for source, kind in ((photographs, "photograph"), (renders, "render")):
        for path, image in _read(source):
            tiles.append(_label(_fit(image), f"{kind}: {path.name}"))
            key.append({"tile": len(key), "kind": kind, "file": str(path)})

    return Sheet(image=_grid(tiles), key=key)


def blind(photographs: list[Path], renders: list[Path], seed: int = 0) -> Sheet:
    """Everything shuffled, numbered, with the answers in a separate file.

    The question to ask over this sheet is not "does this look like him" - that
    invites agreement. It is "are these all the same person, and if not, which
    are the odd ones out".
    """
    entries = [
        (path, image, kind)
        for source, kind in ((photographs, "photograph"), (renders, "render"))
        for path, image in _read(source)
    ]

    random.Random(seed).shuffle(entries)

    tiles, key = [], []
    for index, (path, image, kind) in enumerate(entries, start=1):
        tiles.append(_label(_fit(image), str(index)))
        key.append({"tile": index, "kind": kind, "file": str(path)})

    return Sheet(image=_grid(tiles), key=key)
