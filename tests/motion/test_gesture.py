"""Proof that the body is generated rather than replayed.

The requirement is the one noise.py answers for idle motion, applied to
deliberate movement: "It should be dynamically changing based on emotions,
content the avatar is saying, gestures, etc. It should behave like a human and
not like a loop video."

The tempting way to read that is variety of choice, and most of the tests below
would pass on a scheduler that picked at random from twelve stored clips. The
ones that matter are the first block: two nods in one session are measurably
different nods. A library of clips is a library of loops, and no amount of
shuffling removes them.

After that, the properties that keep the generator usable: the movement joins
stillness without a cut, its apex lands on the syllable it was promised to, and
an event that cannot reach that apex in time is dropped rather than hurried.
"""

import math
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
import pytest

from avatar.motion.gesture import (
    AMPLITUDE_RANGE,
    GESTURES,
    MAX_PER_UTTERANCE,
    MIN_GAP_S,
    NOD_SMALL_RAD,
    PATTERN_LIMIT,
    SAMPLE_DT,
    Curve,
    GestureScheduler,
)
from avatar.motion.pose import BY_NAME

AFFECTS = ("neutral", "amused", "sad", "emphatic", "thoughtful")


def instances(name: str, count: int, seed: int = 0) -> list[Curve]:
    """Curves drawn the way a session draws them: one generator, in sequence."""
    spec = GESTURES[name]
    rng = np.random.default_rng(seed)
    return [spec.generate(spec, rng) for _ in range(count)]


def main_channel(curve: Curve) -> str:
    """The channel carrying the movement, rather than the asymmetry riding it."""
    return max(curve.channels, key=curve.peak)


def beat_count(curve: Curve) -> int:
    """How many separate rise-and-falls the instance has.

    Beats are spaced so the previous one has returned to rest before the next
    peaks, so counting runs above a floor counts beats.
    """
    values = np.abs(curve.channels[main_channel(curve)])
    loud = values > 0.05 * values.max()
    return int(np.count_nonzero(loud[1:] & ~loud[:-1]) + int(loud[0]))


def velocity(curve: Curve, channel: str, times: np.ndarray) -> np.ndarray:
    h = 1e-3
    return (curve.sample(channel, times + h) - curve.sample(channel, times - h)) / (2 * h)


def run(
    session_id,
    affect: str = "neutral",
    count: int = 200,
    spacing: float = 3.0,
    per_utterance: int = 4,
    headroom: float = 5.0,
):
    """A session's worth of accepted events, offered anchors utterance by utterance."""
    scheduler = GestureScheduler(session_id=session_id)
    events = []
    anchor = 0.0

    while len(events) < count:
        scheduler.begin_utterance()
        for k in range(per_utterance):
            t = anchor + k * spacing
            event = scheduler.propose(t, affect=affect, now=t - headroom)
            if event is not None:
                events.append(event)
        anchor += per_utterance * spacing + 2.0

    return scheduler, events[:count]


# -- the generator: where "not a loop video" actually lives -----------------


def test_two_nods_in_one_session_are_not_the_same_nod():
    """The test the whole module exists to pass.

    Asserted on the sampled values rather than on the drawn parameters,
    because a difference in a seed that never reaches the curve is not a
    difference anyone can see.
    """
    first, second = instances("nod_small", 2, seed=0)

    grid = np.linspace(0.0, max(first.duration, second.duration), 600)
    a = first.sample("head_pitch", grid)
    b = second.sample("head_pitch", grid)

    assert np.abs(a - b).max() > 0.2 * np.abs(a).max()


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_no_two_instances_of_a_gesture_are_ever_alike(name):
    """Not just consecutive ones. Forty instances, all distinct, every gesture."""
    curves = instances(name, 40, seed=3)
    span = max(c.duration for c in curves)
    grid = np.linspace(0.0, span, 400)

    traces = [c.sample(main_channel(c), grid) for c in curves]
    scale = max(np.abs(t).max() for t in traces)

    closest = min(
        np.abs(traces[i] - traces[j]).max()
        for i in range(len(traces))
        for j in range(i + 1, len(traces))
    )
    # Half a percent of the largest movement, over 780 pairs. The bar is
    # deliberately far above float noise: two curves that differ only in the
    # sixth decimal are the same curve to a viewer.
    assert closest > 0.005 * scale


def test_a_nod_is_a_family_of_amplitudes_not_an_amplitude():
    """0.6 to 1.4 of nominal, and the whole span used rather than jittered."""
    peaks = np.array([c.peak("head_pitch") for c in instances("nod_small", 400, seed=5)])
    ratio = peaks / NOD_SMALL_RAD

    low, high = AMPLITUDE_RANGE
    assert low - 1e-3 <= ratio.min() < low + 0.05
    assert high - 0.05 < ratio.max() <= high + 1e-3


def test_a_nod_is_one_two_or_three_beats():
    """A fixed beat count is the loudest tell there is: it makes a rhythm."""
    counts = Counter(beat_count(c) for c in instances("nod_small", 400, seed=6))

    assert set(counts) == {1, 2, 3}
    assert min(counts.values()) > 20


def test_a_nod_is_never_purely_sagittal_and_never_leans_the_same_way():
    """A head that nods on one exact axis is a hinge; one that always leans the
    same way off it has a signature, which is a loop of a subtler kind."""
    curves = instances("nod_small", 200, seed=7)
    # The signed extreme, not the maximum: a nod that leans left never has a
    # positive roll at all, and taking the max would read it as having none.
    rolls = np.array(
        [c.channels["head_roll"][np.argmax(np.abs(c.channels["head_roll"]))] for c in curves]
    )

    assert (rolls > 0).any() and (rolls < 0).any()
    assert np.abs(rolls).min() > 0.0


def test_a_nod_rises_in_the_time_a_human_nod_takes():
    """110-180ms from onset to apex, measured rather than assumed."""
    rises = np.array([c.rise_s for c in instances("nod_small", 400, seed=8)])

    assert rises.min() >= 0.110
    assert rises.max() <= 0.180
    assert rises.max() - rises.min() > 0.05


def test_a_movement_always_falls_more_slowly_than_it_rises():
    """Falling as fast as it rose reads as struck rather than as done."""
    for curve in instances("nod_deep", 100, seed=9):
        fall = curve.duration - curve.rise_s if beat_count(curve) == 1 else None
        if fall is not None:
            assert fall > curve.rise_s * 1.2 - 1e-9


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_every_instance_stays_inside_the_declared_pose_ranges(name):
    """A gesture that needs clamping was specified wrong - see pose.py."""
    for curve in instances(name, 200, seed=10):
        for channel, values in curve.channels.items():
            limits = BY_NAME[channel]
            assert values.min() >= limits.low - 1e-9
            assert values.max() <= limits.high + 1e-9


# -- the curve: joining stillness without a cut ----------------------------


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_a_curve_is_silent_outside_its_own_window(name):
    curve = instances(name, 1, seed=11)[0]
    outside = np.concatenate(
        [np.linspace(-2.0, -1e-6, 200), np.linspace(curve.duration + 1e-6, curve.duration + 2.0, 200)]
    )

    for channel in curve.channels:
        assert np.abs(curve.sample(channel, outside)).max() == 0.0


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_a_curve_joins_stillness_with_no_step_in_value_or_velocity(name):
    """The test that catches a visible cut.

    A step in value is an obvious jump. A step in velocity is the subtler one -
    the movement is continuous but starts or stops instantly, which the eye
    reads as a splice even when it cannot say why.
    """
    curve = instances(name, 1, seed=12)[0]
    channel = main_channel(curve)

    span = np.arange(-0.05, curve.duration + 0.05, SAMPLE_DT / 2)
    values = curve.sample(channel, span)
    speed = velocity(curve, channel, span)

    at_edge = (np.abs(span) < 0.004) | (np.abs(span - curve.duration) < 0.004)

    assert np.abs(values[at_edge]).max() < 0.005 * np.abs(values).max()
    assert np.abs(speed[at_edge]).max() < 0.02 * np.abs(speed).max()


def test_the_same_measurement_catches_a_curve_that_is_cut():
    """A test nothing fails is not evidence of anything.

    A ramp that stops at full amplitude is exactly the defect above: continuous
    inside its window, and a hard step to rest at the end of it.
    """
    cut = Curve(channels={"head_pitch": np.linspace(0.0, 0.05, 200)}, dt=SAMPLE_DT, rise_s=0.1)

    span = np.arange(-0.05, cut.duration + 0.05, SAMPLE_DT / 2)
    values = cut.sample("head_pitch", span)
    at_edge = np.abs(span - cut.duration) < 0.004

    assert np.abs(values[at_edge]).max() > 0.5 * np.abs(values).max()


@pytest.mark.parametrize("name", sorted(GESTURES))
def test_the_largest_value_falls_where_the_scheduler_was_promised_it_would(name):
    """rise_s is a promise about the apex, and the scheduler spends it: onset is
    anchor minus this number, so an apex somewhere else is a mistimed gesture."""
    for curve in instances(name, 40, seed=13):
        channel = main_channel(curve)
        grid = np.linspace(0.0, curve.duration, 2000)
        apex = grid[np.argmax(np.abs(curve.sample(channel, grid)))]

        assert apex == pytest.approx(curve.rise_s, abs=0.005)


def test_an_event_reports_its_channels_on_the_speech_timeline():
    scheduler = GestureScheduler(session_id="timeline")
    event = None
    anchor = 10.0
    while event is None:
        scheduler.begin_utterance()
        event = scheduler.propose(anchor, now=0.0)
        anchor += 5.0
    anchor = event.anchor

    channel = main_channel(event.curve)
    grid = np.linspace(event.onset - 0.5, event.end + 0.5, 4000)
    peak = grid[np.argmax([abs(event.at(t)[channel]) for t in grid])]

    assert peak == pytest.approx(anchor, abs=0.01)
    assert event.at(event.onset - 0.1)[channel] == 0.0
    assert event.at(event.end + 0.1)[channel] == 0.0


# -- the scheduler ---------------------------------------------------------


def test_no_gesture_repeats_inside_its_own_cooldown():
    _, events = run("cooldown", count=200)
    last: dict[str, float] = {}

    for event in events:
        if event.name in last:
            assert event.anchor - last[event.name] >= GESTURES[event.name].cooldown_s
        last[event.name] = event.anchor


def test_no_three_gesture_sequence_happens_more_than_twice():
    """A loop is an order, not a tally.

    The recency weighting cannot see this - it counts single gestures - so the
    scheduler vetoes the third occurrence of a triple outright. Chance alone
    would produce four and five at this length.
    """
    _, events = run("patterns", count=200)
    names = [e.name for e in events]

    triples = Counter(tuple(names[i : i + 3]) for i in range(len(names) - 2))
    assert max(triples.values()) <= PATTERN_LIMIT


def test_the_choice_of_gesture_stays_above_two_bits_of_entropy():
    """Seven gestures is 2.81 bits at best. Anything near one bit is a habit."""
    _, events = run("entropy", count=200)
    counts = Counter(e.name for e in events)

    share = [n / len(events) for n in counts.values()]
    entropy = -sum(p * math.log2(p) for p in share)

    assert entropy > 2.0


def test_one_session_replays_exactly():
    """So that 'the body did something odd at 40 seconds' can be looked at."""
    _, first = run("replay-me", count=60)
    _, second = run("replay-me", count=60)

    assert [(e.name, e.onset) for e in first] == [(e.name, e.onset) for e in second]
    for a, b in zip(first, second, strict=True):
        channel = main_channel(a.curve)
        assert np.array_equal(a.curve.channels[channel], b.curve.channels[channel])


def test_two_sessions_do_not_gesture_in_the_same_order():
    _, a = run("session-one", count=60)
    _, b = run("session-two", count=60)

    assert [e.name for e in a] != [e.name for e in b]


def test_an_event_without_room_to_reach_its_apex_is_dropped():
    """Not compressed, not started late. A nod that lands after the stressed
    syllable is a different gesture from the one that was chosen."""
    scheduler = GestureScheduler(session_id="no-headroom")

    for i in range(200):
        scheduler.begin_utterance()
        anchor = i * 3.0 + 0.05
        assert scheduler.propose(anchor, now=anchor - 0.01) is None

    assert scheduler.dropped > 0


def test_the_same_anchors_are_taken_when_there_is_room():
    """The drop above is about headroom and nothing else.

    Same session, same anchors, only now moved back: the counter goes to zero
    and the events appear. Without this the test above would also pass on a
    scheduler that had simply stopped working.
    """
    scheduler = GestureScheduler(session_id="no-headroom")
    taken = 0

    for i in range(200):
        scheduler.begin_utterance()
        anchor = i * 3.0 + 0.05
        if scheduler.propose(anchor, now=anchor - 5.0) is not None:
            taken += 1

    assert taken > 50
    assert scheduler.dropped == 0


@pytest.mark.parametrize("affect", AFFECTS)
def test_no_event_ever_starts_later_than_its_anchor_minus_its_rise(affect):
    """The scheduling rule, asserted mechanically over a whole session."""
    _, events = run(f"onsets-{affect}", affect=affect, count=150)

    for event in events:
        assert event.onset <= event.anchor - event.curve.rise_s + 1e-9
        assert event.onset == pytest.approx(event.anchor - event.curve.rise_s)


def test_at_most_three_body_actions_in_one_utterance():
    """However many anchors the utterance offers. Gesturing on every clause
    reads as a puppet, and reads that way faster than gesturing rarely."""
    scheduler = GestureScheduler(session_id="budget")

    for utterance in range(40):
        scheduler.begin_utterance()
        base = utterance * 200.0
        fired = [
            scheduler.propose(base + k * MIN_GAP_S, now=0.0)
            for k in range(20)
        ]
        assert sum(e is not None for e in fired) <= MAX_PER_UTTERANCE


def test_at_most_one_body_action_per_two_and_a_half_seconds():
    _, events = run("spacing", count=200, spacing=1.0, per_utterance=8)

    gaps = [b.anchor - a.anchor for a, b in pairwise(events)]
    assert min(gaps) >= MIN_GAP_S - 1e-9


def test_two_events_never_own_the_same_region_at_once():
    """The head cannot be nodding and shaking; the montages would fight."""
    _, events = run("regions", count=200, spacing=2.6)

    by_region = defaultdict(list)
    for event in events:
        by_region[event.region].append(event)

    for region, occupants in by_region.items():
        occupants.sort(key=lambda e: e.onset)
        for earlier, later in pairwise(occupants):
            assert later.onset >= earlier.end - 1e-9, region


def test_sadness_and_amusement_move_the_body_differently():
    """Affinity has to reach the output, not just the weights: cooldowns cap how
    often anything can fire, so a mood that only tilts a weight changes nothing."""

    def distribution(affect: str) -> dict[str, float]:
        counts: Counter[str] = Counter()
        for session in range(12):
            _, events = run(f"{affect}-{session}", affect=affect, count=120)
            counts.update(e.name for e in events)
        total = sum(counts.values())
        return {name: counts[name] / total for name in GESTURES}

    sad = distribution("sad")
    amused = distribution("amused")

    divergence = 0.5 * sum(abs(sad[n] - amused[n]) for n in GESTURES)
    assert divergence > 0.05

    # Direction, not just magnitude: a shrug is an amused gesture and a deep nod
    # is not, and the distributions have to agree.
    assert amused["shrug"] > sad["shrug"] * 1.2
    assert sad["nod_deep"] > amused["nod_deep"] * 1.2


def test_an_unknown_affect_label_is_treated_as_neutral():
    """A new label from the director must degrade to plain, never to stillness."""
    scheduler = GestureScheduler(session_id="unknown")
    spec = GESTURES["nod_small"]

    known = scheduler.effective_weight(spec, anchor=100.0, affect="neutral")
    unknown = scheduler.effective_weight(spec, anchor=100.0, affect="wistful")

    assert unknown == known == spec.base_weight


def test_the_body_declines_far_more_often_than_it_agrees():
    """Under-gesturing beats over-gesturing. Every anchor is an offer, and most
    of them have to be refused for the likeness not to read as a puppet."""
    scheduler = GestureScheduler(session_id="restraint")
    offered = taken = 0

    for utterance in range(200):
        scheduler.begin_utterance()
        base = utterance * 30.0
        for k in range(8):
            anchor = base + k * 1.0
            offered += 1
            if scheduler.propose(anchor, now=anchor - 5.0) is not None:
                taken += 1

    assert taken / offered < 0.4
