"""The browser and the sender must agree on what each number means.

A pose frame is 35 floats with no names on the wire. If the two sides disagree
about the order, nothing errors: every value is a valid float in a valid slot,
and the face simply does the wrong thing. The head turns when it should blink.

That already happened once here, and only the first three channels were live at
the time, so it would have sat undetected until the fourth was wired. These
tests read the browser's own file and compare it to the Python definition,
because a comment saying they match is not the same as them matching.
"""

import re
from pathlib import Path

from avatar.motion.pose import VISEME_COUNT, VISEME_NAMES, channel_names

WIRE = Path(__file__).resolve().parents[2] / "apps" / "web" / "lib" / "pose.ts"


def _array(source: str, name: str) -> list[str]:
    match = re.search(rf"{name}[^=]*=\s*\[(.*?)\n\] as const;", source, re.DOTALL)
    assert match, f"{name} not found in {WIRE}"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_the_browser_reads_the_channels_in_the_order_the_sender_writes_them():
    """The bug this catches is silent: every value lands in a real slot."""
    assert _array(WIRE.read_text(), "POSE_CHANNELS") == channel_names()


def test_the_browser_reads_the_visemes_in_the_order_the_sender_writes_them():
    assert _array(WIRE.read_text(), "POSE_VISEMES") == list(VISEME_NAMES)


def test_the_frame_is_the_size_both_sides_expect():
    """8 bytes of header, then two float arrays. A different length is dropped
    whole on arrival, so a size drift shows up as a call with no motion at all
    rather than as an error."""
    source = WIRE.read_text()
    expected = 8 + 4 * len(channel_names()) + 4 * VISEME_COUNT

    assert str(expected) in source, f"{expected} bytes not stated in {WIRE.name}"
