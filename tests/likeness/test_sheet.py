"""The likeness sheet.

Gate 1 is decided by a person looking at this, so what matters is that the
sheet does not tell them the answer. Every test here is about that: no
distortion that changes a face, no ragged edge that marks the last tile, and
the answer key written somewhere other than the picture.
"""

import json

import cv2
import numpy as np
import pytest

from avatar.likeness.sheet import COLUMNS, TILE, blind, comparison


@pytest.fixture
def images(tmp_path):
    def write(name, value, size):
        path = tmp_path / name
        cv2.imwrite(str(path), np.full((*size, 3), value, dtype=np.uint8))
        return path

    photographs = [write(f"photo-{i}.jpg", 40 + i * 10, (600, 400)) for i in range(3)]
    renders = [write(f"render-{i}.png", 200 - i * 10, (512, 512)) for i in range(2)]
    return photographs, renders


def test_a_comparison_sheet_holds_every_image(images):
    photographs, renders = images
    sheet = comparison(photographs, renders)

    assert len(sheet.key) == len(photographs) + len(renders)
    assert {e["kind"] for e in sheet.key} == {"photograph", "render"}


def test_a_blind_sheet_does_not_reveal_which_is_which(images):
    """A labelled sheet collects politeness, not judgement."""
    photographs, renders = images
    sheet = blind(photographs, renders, seed=1)

    kinds = [e["kind"] for e in sheet.key]
    # Shuffled: the renders are not simply the last tiles.
    assert kinds != sorted(kinds, key=lambda k: k == "render")
    assert [e["tile"] for e in sheet.key] == list(range(1, len(kinds) + 1))


def test_the_answer_key_is_written_beside_the_sheet_not_into_it(images, tmp_path):
    photographs, renders = images
    path = blind(photographs, renders).save(tmp_path / "sheet.png")

    key = json.loads(path.with_suffix(".key.json").read_text())

    assert path.exists()
    assert len(key) == 5
    assert all("kind" in e for e in key)


def test_images_are_letterboxed_rather_than_stretched(images):
    """A face squeezed by an aspect change is a different face."""
    photographs, renders = images
    sheet = comparison(photographs[:1], renders[:1])

    # Two tiles, one column each on a COLUMNS-wide grid: the sheet is a whole
    # number of tiles wide however few images were given.
    assert sheet.image.shape[1] == COLUMNS * (TILE + 12)


def test_a_short_last_row_is_padded(images):
    """A ragged edge points at the final tile, which on a blind sheet is a hint."""
    photographs, renders = images
    sheet = blind(photographs, renders)

    rows = -(-len(sheet.key) // COLUMNS)
    assert sheet.image.shape[0] == rows * (TILE + 28 + 12)


def test_unreadable_images_are_reported_not_silently_dropped(tmp_path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_text("nope")

    with pytest.raises(ValueError, match="could be read"):
        comparison([bad], [])
