"""Proof that nothing the model does to its own tag reaches the voice.

The affect tag is a contract with a small model, and a small model breaks it.
It truncates it, doubles it, invents a label, puts a greeting above it. Every
one of those is a normal Tuesday, and every one of them has the same
consequence if it is not caught: a synthesiser reading "[[warm|0.6]]" aloud to
somebody who is talking to a recreation of their mother. So most of this file
is about text rather than about feelings.

The rest defends the rate limit, which is the difference between a face and an
interface, and the single documented hole in it.
"""

import time

import pytest

from avatar.motion.affect import (
    AFFECT_LABELS,
    CRISIS_RATES,
    CRISIS_REACH_S,
    RATES,
    Affect,
    Rates,
    affect_for,
    affect_from_label,
    blend,
    lexicon_affect,
    parse_tag,
)

REST_VALENCE, REST_AROUSAL, _ = AFFECT_LABELS["neutral"]


def at(label: str, intensity: float = 1.0) -> Affect:
    affect = affect_from_label(label, intensity, confidence=1.0, source="tag")
    assert affect is not None
    return affect


def arrived(current: Affect, target: Affect, tolerance: float = 1e-9) -> bool:
    return (
        abs(current.valence - target.valence) <= tolerance
        and abs(current.arousal - target.arousal) <= tolerance
        and abs(current.dominance - target.dominance) <= tolerance
    )


def seconds_to_reach(
    previous: Affect, target: Affect, *, rates: Rates = RATES, fps: float = 200.0
) -> float:
    """How long the face takes to actually get there, stepped like a renderer.

    Stepped rather than solved so it measures what blend() does per frame, not
    what the rate table says it should do.
    """
    dt = 1.0 / fps
    elapsed = 0.0
    current = previous
    while elapsed < 10.0:
        current = blend(current, target, dt, rates=rates)
        elapsed += dt
        if arrived(current, target):
            return elapsed
    return float("inf")


# --------------------------------------------------------------------------
# The tag


def test_a_well_formed_tag_is_read_and_removed_from_the_spoken_text():
    affect, clean = parse_tag("[[warm|0.6]]\nI was hoping you would call.")

    assert clean == "I was hoping you would call."
    assert affect is not None
    assert affect.label == "warm"
    assert affect.intensity == 0.6
    assert affect.source == "tag"


@pytest.mark.parametrize(
    "spelling",
    [
        "[[warm|0.6]]",
        "[[ warm | 0.6 ]]",
        "[[Warm|0.6]]",
        "[[WARM|0.6]]",
        "[[warm|.6]]",
        "  [[warm|0.6]]  ",
    ],
)
def test_spacing_and_case_do_not_cost_a_correct_reading(spelling):
    """The model gets the punctuation wrong far more often than the feeling."""
    affect, clean = parse_tag(f"{spelling}\nHello.")

    assert clean == "Hello."
    assert affect is not None and affect.label == "warm"


@pytest.mark.parametrize(
    "mangled",
    [
        "[[warm|0.6",
        "[warm|0.6]]",
        "[[warm 0.6]]",
        "[[|0.6]]",
        "[[warm|]]",
        "[[warm]]",
        "[[warm|banana]]",
        "[[warm|nan]]",
        "[[warm|inf]]",
        "[[warm|1e400]]",
        "[[warm|0.6]]]]",
        "[[[[warm|0.6]]",
        "[[warm|0.6|0.2]]",
        "[[warm|-2]]",
        "[[]]",
        "[[",
    ],
)
def test_a_mangled_tag_never_raises_and_never_reaches_the_voice(mangled):
    """Removal is not best effort, whatever parsing decided.

    Each input here is the whole of the model's reply, so anything left over is
    something a synthesiser would say out loud.
    """
    _, clean = parse_tag(mangled)

    assert clean == ""


def test_an_absent_tag_leaves_the_reply_exactly_as_it_was():
    reply = "I never did learn to drive.\nYour father tried to teach me."

    assert parse_tag(reply) == (None, reply)


@pytest.mark.parametrize(
    "reply",
    [
        "He wrote [[see the letter]] in the margin.",
        "The array is [[1, 2], [3, 4]] in the notebook.",
        "[[unknown|thing]] is what the file said.",
        "[[Chapter One]]\nIt began in the spring.",
        "She used [[ and ]] to mark her own asides.",
    ],
)
def test_double_brackets_that_are_not_a_tag_survive_untouched(reply):
    """The parser is allowed to be silent, not to edit somebody's sentence."""
    affect, clean = parse_tag(reply)

    assert clean == reply
    assert affect is None


def test_an_unknown_label_falls_back_rather_than_inventing_coordinates():
    """A feeling nobody defined has no place on the face to be."""
    affect, clean = parse_tag("[[excited|0.8]]\nThey are all coming on Sunday.")

    assert affect is None
    assert clean == "They are all coming on Sunday."


def test_affect_from_label_refuses_a_label_outside_the_table():
    assert affect_from_label("wistful", 0.5, confidence=0.9, source="tag") is None


def test_a_tag_emitted_twice_is_read_once_and_removed_completely():
    affect, clean = parse_tag("[[sad|0.7]]\n[[sad|0.7]]\nI think about it often.")

    assert clean == "I think about it often."
    assert affect is not None and affect.label == "sad"


def test_a_tag_below_the_first_line_is_still_taken_off_the_text():
    affect, clean = parse_tag("Hello, love.\n[[warm|0.5]]\nCome and sit down.")

    assert "[[" not in clean
    assert clean == "Hello, love.\n\nCome and sit down."
    assert affect is not None and affect.label == "warm"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("[[warm|1.7]]", 1.0), ("[[warm|-0.4]]", 0.0), ("[[warm|1.0]]", 1.0)],
)
def test_an_out_of_range_intensity_is_clamped_rather_than_thrown_away(raw, expected):
    """The label is still information even when the number is nonsense."""
    affect, _ = parse_tag(raw)

    assert affect is not None
    assert affect.intensity == expected


def test_a_non_finite_intensity_never_becomes_a_face():
    """One NaN in the rate limiter freezes the face silently for the whole call."""
    affect, clean = parse_tag("[[warm|nan]]\nGo on then.")

    assert affect is None
    assert clean == "Go on then."


# --------------------------------------------------------------------------
# Intensity


def test_intensity_scales_away_from_rest_and_not_from_zero():
    half = at("sad", 0.5)

    assert half.valence == pytest.approx(REST_VALENCE + (-0.60 - REST_VALENCE) * 0.5)
    assert half.arousal == pytest.approx(0.275)
    # Scaled from zero this would be 0.10, well under a resting face: a
    # half-strength sadness would render as somebody falling asleep.
    assert half.arousal > 0.1


def test_full_intensity_lands_exactly_on_the_declared_coordinates():
    for label, (valence, arousal, dominance) in AFFECT_LABELS.items():
        full = at(label, 1.0)

        assert (full.valence, full.arousal, full.dominance) == pytest.approx(
            (valence, arousal, dominance)
        )


def test_dominance_is_a_stance_and_does_not_fade_with_intensity():
    """A quiet grief and a heavy grief are equally low-dominance."""
    assert at("sad", 0.2).dominance == at("sad", 1.0).dominance


def test_intensity_is_rounded_so_a_retry_produces_the_same_face():
    affect, _ = parse_tag("[[warm|0.6499]]")

    assert affect is not None and affect.intensity == 0.6


# --------------------------------------------------------------------------
# The fallback


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("I love you, sweetheart.", "warm"),
        ("That is absolutely hilarious.", "amused"),
        ("She died in the winter and I still cry about it.", "sad"),
        ("Maybe. Perhaps I am misremembering.", "uncertain"),
        ("Listen carefully, this is important.", "serious"),
        ("Hold my hand, gently.", "tender"),
        ("Oh. Wow.", "surprised"),
        ("The train leaves at four.", "neutral"),
    ],
)
def test_the_lexicon_gets_the_obvious_cases(reply, expected):
    """Only the obvious ones. It is a word list and it is wrong constantly."""
    assert lexicon_affect(reply).label == expected


def test_the_lexicon_says_it_is_guessing():
    affect = lexicon_affect("I love you.")

    assert affect.source == "lexicon"
    assert affect.confidence == pytest.approx(0.5)


def test_a_reply_with_no_emotional_words_rests_at_neutral():
    affect = lexicon_affect("The train leaves at four.")

    assert affect.label == "neutral"
    assert (affect.valence, affect.arousal) == pytest.approx(
        (REST_VALENCE, REST_AROUSAL)
    )


@pytest.mark.parametrize(
    "reply",
    ["", "   ", "\n\n", "...", "?!", "-", "]]", "x" * 500_000, "sad " * 50_000],
)
def test_the_lexicon_never_raises_on_anything_a_model_can_return(reply):
    assert lexicon_affect(reply).label in AFFECT_LABELS


def test_the_lexicon_is_cheap_enough_to_run_on_every_turn():
    """It sits between the model returning and the voice starting.

    Anything expensive there is paid for in silence the caller can hear, so the
    long input is capped rather than scanned.
    """
    reply = "I remember that summer. " * 400

    start = time.perf_counter()
    for _ in range(200):
        lexicon_affect(reply)
    per_call = (time.perf_counter() - start) / 200

    assert per_call < 0.001


def test_one_word_repeated_does_not_outvote_several_that_agree():
    assert lexicon_affect("sad sad sad sad sad, but I laugh and joke and grin").label == "amused"


# --------------------------------------------------------------------------
# The pipeline entry point


def test_affect_for_prefers_the_model_over_the_word_list():
    """The model knows it is being gentle. The word list never will."""
    affect, clean = affect_for("[[serious|0.8]]\nI love you, but you must listen.")

    assert affect.label == "serious"
    assert affect.source == "tag"
    assert clean == "I love you, but you must listen."


def test_affect_for_falls_through_to_the_word_list_when_the_tag_is_missing():
    affect, clean = affect_for("I love you, sweetheart.")

    assert affect.label == "warm"
    assert affect.source == "lexicon"
    assert clean == "I love you, sweetheart."


@pytest.mark.parametrize(
    "reply",
    ["", "   ", None, "[[", "[[warm|0.6", "[[warm|nan]]", "\n\n\n", "]]" * 1000],
)
def test_affect_for_always_returns_a_face_and_speakable_text(reply):
    """None is in this list on purpose: a model client can return one."""
    affect, clean = affect_for(reply)

    assert affect.label in AFFECT_LABELS
    assert isinstance(clean, str)


# --------------------------------------------------------------------------
# Getting there


def test_a_full_valence_swing_takes_the_seconds_the_limit_implies():
    """The limit is 1.2 per second, so 1.2 of valence is one second. Exactly."""
    low = Affect(-0.6, 0.35, 0.5, "sad", 1.0, 0.9, "tag")
    high = Affect(0.6, 0.35, 0.5, "warm", 1.0, 0.9, "tag")

    assert seconds_to_reach(low, high) == pytest.approx(1.0, abs=0.01)


def test_amusement_to_gravity_takes_about_seven_hundred_milliseconds():
    """The number the whole design is arranged around.

    A face that flips instantly reads as an interface. This delay is not
    latency to be optimised away.
    """
    assert seconds_to_reach(at("amused"), at("serious")) == pytest.approx(0.71, abs=0.02)


def test_the_face_never_arrives_in_a_single_frame():
    one_frame = blend(at("amused"), at("sad"), 1.0 / 25.0)

    assert not arrived(one_frame, at("sad"))
    assert one_frame.valence == pytest.approx(0.70 - 1.2 / 25.0)


def test_blending_reports_the_destination_while_still_travelling():
    """The label is the intent, the coordinates are the face. They disagree."""
    moving = blend(at("amused"), at("serious"), 1.0 / 25.0)

    assert moving.label == "serious"
    assert moving.valence > 0.5


def test_blending_never_overshoots():
    current = at("sad")
    for _ in range(10):
        current = blend(current, at("amused"), 0.5)

    assert arrived(current, at("amused"))


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_a_nonsense_timestep_leaves_the_face_where_it_was(dt):
    previous = at("warm")

    assert blend(previous, at("sad"), dt) == previous


# --------------------------------------------------------------------------
# The one exception


@pytest.mark.parametrize("label", sorted(AFFECT_LABELS))
def test_the_crisis_override_reaches_serious_inside_the_budget(label):
    """The crisis check short-circuits the turn and a fixed message is spoken.

    The face has to be serious when that sentence starts, not easing into it
    over most of a second.
    """
    reached = seconds_to_reach(at(label), at("serious"), rates=CRISIS_RATES)

    assert reached <= CRISIS_REACH_S


def test_the_crisis_override_is_still_a_movement_and_not_a_cut():
    """One frame is a screen changing state, at the worst possible moment."""
    one_frame = blend(at("amused"), at("serious"), 1.0 / 25.0, rates=CRISIS_RATES)

    assert not arrived(one_frame, at("serious"))


@pytest.mark.parametrize(
    "label", sorted(set(AFFECT_LABELS) - {"serious"})
)
def test_nothing_but_the_crisis_override_reaches_serious_that_fast(label):
    """Ordinary rates cannot do it from any feeling the face can be in.

    "serious" itself is excluded because a face already there has nowhere to
    travel, which is arrival rather than a bypass.
    """
    reached = seconds_to_reach(at(label), at("serious"))

    assert reached > CRISIS_REACH_S
