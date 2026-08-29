"""What the face means, as distinct from what the mouth is doing.

Visemes come from the audio and the head drifts on its own - see noise.py -
and both run without knowing anything about the reply. This module supplies the
missing third thing: whether the person saying it is amused, or careful, or has
just been asked something they cannot answer. Without it the likeness is a
correctly lip-synced mask, which is a specific and well documented way to be
unsettling.

Emotion arrives on two paths and they are not equally good. The language model
is asked to declare its own intent on the first line of the reply, which is by
far the better signal: the model knows it is being gentle and a word list never
will. It is also the less reliable one, because a small model omits or mangles
that line often enough that the fallback is not a nicety. So there is a word
list too, it is honestly mediocre, and it is always there.

Everything here is deliberately cheap. It runs between the model returning and
the voice starting, on the same turn, and anything expensive at that point is
paid for in silence the caller can hear.

The last piece is rate limiting. A face that changes expression in one frame
does not read as a person changing their mind, it reads as an interface
updating. Movement between affects is held to roughly the speed a face can
actually move, which puts about 700ms between amusement and gravity. That delay
is not latency to be optimised away; it is the thing that makes it look alive.
The single exception is written into the signature rather than hidden behind a
flag, because it is a safety requirement and not a preference: when the crisis
check fires, the face has to be serious before the sentence starts.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# The closed set. Valence is pleasant/unpleasant, arousal is activation,
# dominance is how much the speaker owns the moment. Eight labels rather than a
# free-text mood because a face has to be built from these numbers, and a model
# allowed to invent "wistful" produces a coordinate nobody has defined.
AFFECT_LABELS: dict[str, tuple[float, float, float]] = {
    "neutral": (0.00, 0.35, 0.50),
    "warm": (0.55, 0.45, 0.45),
    "tender": (0.35, 0.25, 0.30),
    "sad": (-0.60, 0.20, 0.30),
    "amused": (0.70, 0.70, 0.55),
    "serious": (-0.15, 0.40, 0.70),
    "uncertain": (-0.10, 0.35, 0.25),
    "surprised": (0.10, 0.80, 0.40),
}

# Where a face sits when nothing is happening. Intensity moves valence and
# arousal between this point and the label's own coordinates rather than
# between zero and them: scaled from zero, a half-strength sadness would land
# at arousal 0.10, which is below a resting face. Half a feeling is not half
# asleep.
REST_VALENCE, REST_AROUSAL = AFFECT_LABELS["neutral"][0], AFFECT_LABELS["neutral"][1]

# Where an affect came from. "carry" is written by the director when a turn
# produces no evidence at all and the previous affect is simply held.
SOURCES = ("tag", "lexicon", "carry")

# The model declaring its own intent is worth far more than a word count, but
# it is not worth everything: it is still a small model, and it is wrong about
# its own tone often enough that a downstream consumer should be able to weigh
# the two apart.
TAG_CONFIDENCE = 0.9
LEXICON_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Affect:
    """A point in valence/arousal/dominance space, and where it came from.

    The three numbers are what the face shows. The label, intensity and source
    are provenance, and they are allowed to disagree with the numbers: while
    blend() is moving the face towards a new feeling, the label already names
    the destination and the coordinates are still somewhere on the way. That
    gap is the entire point of the rate limiter, so it is not an inconsistency
    to be normalised away.
    """

    valence: float
    arousal: float
    dominance: float
    label: str
    intensity: float
    confidence: float
    source: str


def _clamp01(value: float) -> float:
    """Non-finite values included, deliberately.

    "nan" and "inf" are both accepted by float(), and a single NaN reaching the
    rate limiter would poison every comparison after it and leave the face
    frozen for the rest of the call without ever raising anything.
    """
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _at(label: str, intensity: float, *, confidence: float, source: str) -> Affect:
    valence, arousal, dominance = AFFECT_LABELS[label]

    # Rounded because the model's own contract is one decimal. It also means
    # the same reply produces the same face on a retry, which matters when
    # somebody reports that the recreation looked wrong at a particular line.
    strength = round(_clamp01(intensity), 1)

    return Affect(
        valence=REST_VALENCE + (valence - REST_VALENCE) * strength,
        arousal=REST_AROUSAL + (arousal - REST_AROUSAL) * strength,
        # Dominance is not scaled. It is a stance rather than an amplitude - a
        # quiet grief and a heavy grief are equally low-dominance - and fading
        # it towards the resting value would make every mild feeling read as
        # the same self-possessed neutral.
        dominance=dominance,
        label=label,
        intensity=strength,
        confidence=confidence,
        source=source,
    )


def affect_from_label(
    label: str, intensity: float, *, confidence: float, source: str
) -> Affect | None:
    """None for a label outside the table.

    The caller falls back rather than receiving invented coordinates. A model
    that emits "excited" has told us something, but not something this system
    can render, and guessing at where excitement sits produces a face nobody
    designed.
    """
    if label not in AFFECT_LABELS:
        return None
    return _at(label, intensity, confidence=confidence, source=source)


# --------------------------------------------------------------------------
# Parsing what the model emitted


# The shape asked for is "[[warm|0.6]]" on the first line. This is permissive
# about case, spacing and the missing intensity, because a small model gets the
# punctuation wrong far more often than it gets the feeling wrong and refusing
# "[[ Warm | 0.6 ]]" throws away a correct reading over a space.
_TAG = re.compile(r"\[\[[ \t]*([A-Za-z_]+)[ \t]*(?:\|[ \t]*([^\[\]|]*?)[ \t]*)?\]\]")

# What counts as an intensity. Plain decimals only - float() also accepts "nan"
# and "inf", and neither is a feeling.
_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)$")

# Everything a mangled tag can decay into: brackets, a pipe, a number, and at
# most one label word.
_TAG_JUNK = r"[\[\]|.,\d \t+-]"
_TAG_RESIDUE = re.compile(
    rf"^{_TAG_JUNK}*(?:{'|'.join(sorted(AFFECT_LABELS))})?{_TAG_JUNK}*$",
    re.IGNORECASE,
)


def _drop_tag_residue(text: str) -> str:
    """Remove a first line that is a tag the model never finished.

    A truncated "[[warm|0.6" matches nothing and would otherwise survive into
    the speech path, where a synthesiser reads it aloud as "warm zero point
    six". That is worse than any wrong expression, so a first line made only of
    brackets, digits and at most one label word is discarded whole. The bracket
    requirement is what protects a real reply that happens to open with a year.
    """
    head, newline, rest = text.partition("\n")
    stripped = head.strip()

    if "[" not in stripped and "]" not in stripped:
        return text
    if _TAG_RESIDUE.fullmatch(stripped):
        return rest if newline else ""
    return text


def parse_tag(text: str) -> tuple[Affect | None, str]:
    """The declared affect and the text with every trace of the tag removed.

    Two separate jobs, and only one of them is allowed to fail. Reading the
    affect is best effort and returns None when the tag is absent or unusable.
    Removing it is not optional: nothing shaped like a tag may reach the voice,
    whether or not it parsed, whether or not the label was one we know, and
    however many times the model emitted it.

    The affect is taken from the first usable tag wherever it appears rather
    than from the first line only. Once removal is unconditional, insisting on
    the line number buys nothing and costs a correct reading every time the
    model puts a greeting above its own tag.
    """
    if not text:
        return None, ""

    found: Affect | None = None

    def strip(match: re.Match[str]) -> str:
        nonlocal found
        label = match.group(1).lower()
        raw = match.group(2) or ""
        numeric = bool(_NUMBER.match(raw))

        # Neither the shape of a tag nor a word from the table: "[[see|note]]"
        # is somebody's own text and none of our business.
        if not numeric and label not in AFFECT_LABELS:
            return match.group(0)

        if found is None and numeric:
            found = affect_from_label(
                label, float(raw), confidence=TAG_CONFIDENCE, source="tag"
            )
        return ""

    clean = _drop_tag_residue(_TAG.sub(strip, text))

    # Only the tag itself is removed. The spaces it leaves behind mid-sentence
    # are left alone, because collapsing whitespace would edit text nobody
    # asked us to touch and a synthesiser cannot hear the difference.
    return found, clean.strip()


# --------------------------------------------------------------------------
# The fallback


# Only the opening of a reply is examined. Replies are one to three sentences
# by design - see persona.py - so this changes nothing in practice, and it caps
# the cost when a model ignores that instruction and returns an essay.
LEXICON_SCAN_CHARS = 2000

_WORDS = re.compile(r"[a-z']+")

# Declaration order is the tie-break, and it runs from the most restrained
# reading to the least. This is a product about grief: mistaking sorrow for
# amusement is the failure somebody remembers, and mistaking amusement for
# sorrow is one they forgive.
_LEXICON: dict[str, frozenset[str]] = {
    "sad": frozenset((
        "sad", "sadly", "sorrow", "sorrowful", "grief", "grieving", "grieved", "mourn",
        "mourning", "cry", "cried", "crying", "tears", "tearful", "lonely", "alone", "loss",
        "lost", "gone", "died", "death", "dying", "hurt", "hurts", "ache", "aching", "empty",
        "heartbroken", "heartbreak", "unbearable", "sorry", "weep", "buried", "funeral"
    )),
    "serious": frozenset((
        "serious", "seriously", "important", "listen", "careful", "carefully", "danger",
        "dangerous", "warn", "warning", "understand", "attention", "consequences",
        "responsibility", "mistake", "truth", "promise", "urgent", "gravely", "firmly"
    )),
    "tender": frozenset((
        "gentle", "gently", "softly", "soft", "quiet", "quietly", "tender", "tenderly",
        "precious", "sweet", "sweetly", "hold", "held", "holding", "miss", "missed", "cherish",
        "safe", "calm", "hush"
    )),
    "uncertain": frozenset((
        "maybe", "perhaps", "might", "unsure", "suppose", "guess", "guessing", "wonder",
        "wondering", "unclear", "possibly", "probably", "doubt", "doubtful", "uncertain",
        "forget", "forgot", "forgotten", "vague", "hazy", "somehow", "somewhere", "seems",
        "dunno", "hmm"
    )),
    "warm": frozenset((
        "love", "loved", "loving", "dear", "darling", "sweetheart", "glad", "proud", "thank",
        "thanks", "thankful", "grateful", "kind", "kindness", "welcome", "happy", "happiness",
        "comfort", "comforting", "hug", "warm", "warmth", "care", "blessed", "wonderful",
        "lovely", "joy", "delighted", "fond"
    )),
    "amused": frozenset((
        "funny", "laugh", "laughed", "laughing", "laughter", "joke", "jokes", "joking",
        "hilarious", "silly", "ridiculous", "haha", "hahaha", "lol", "tease", "teasing",
        "grin", "grinning", "chuckle", "chuckling", "absurd", "mischief", "cheeky"
    )),
    "surprised": frozenset((
        "oh", "wow", "whoa", "goodness", "suddenly", "unexpected", "unexpectedly", "surprise",
        "surprised", "surprising", "astonished", "amazed", "astounded", "incredible",
        "unbelievable", "gosh", "startled"
    )),
}


def lexicon_affect(text: str) -> Affect:
    """A guess from word lists. Always available, frequently wrong.

    Right roughly two thirds of the time on ordinary conversational replies,
    and that is worth having only because it costs nothing and is never
    unavailable. It has no notion of negation, sarcasm, quotation or context:
    "I am not sad at all" reads as sad, and it always will. It is English only,
    so a Spanish reply rests at neutral - a real gap in a bilingual product,
    recorded here rather than papered over.

    The tag is the signal. This is what the face falls back to when the model
    forgets to emit one, and the confidence it reports says so.
    """
    if not text:
        return _at("neutral", 0.0, confidence=LEXICON_CONFIDENCE, source="lexicon")

    # Sliced before lowercasing, so a runaway reply costs the same as a short
    # one instead of paying to fold the case of text nobody will read.
    tokens = set(_WORDS.findall(text[:LEXICON_SCAN_CHARS].lower()))

    label, best = "neutral", 0
    for candidate, words in _LEXICON.items():
        # Distinct words rather than occurrences, so one word repeated through
        # a long reply cannot outvote three different words that agree.
        hits = len(tokens & words)
        if hits > best:
            label, best = candidate, hits

    if best == 0:
        return _at("neutral", 0.0, confidence=LEXICON_CONFIDENCE, source="lexicon")

    # One matching word is weak evidence and should move the face a little; the
    # ceiling is reached at four, past which a word count stops meaning more.
    return _at(
        label, 0.3 + 0.2 * best, confidence=LEXICON_CONFIDENCE, source="lexicon"
    )


def affect_for(text: str) -> tuple[Affect, str]:
    """What the pipeline calls: an affect, and text safe to speak.

    Always returns both. There is no path here that leaves the caller deciding
    what to do about a missing feeling, because the answer is always the same -
    show the resting face - and spreading that decision across call sites is
    how one of them ends up showing nothing at all.
    """
    try:
        tagged, clean = parse_tag(text)
        if tagged is not None:
            return tagged, clean
        return lexicon_affect(clean), clean
    except Exception:  # noqa: BLE001 - a face is never worth dropping a call for
        # The last thing between the model and the voice on a live call.
        # Nothing about an expression is worth dropping a call for, so an input
        # nobody anticipated produces a resting face and untouched text.
        resting = _at("neutral", 0.0, confidence=LEXICON_CONFIDENCE, source="lexicon")
        return resting, text or ""


# --------------------------------------------------------------------------
# Getting there


@dataclass(frozen=True)
class Rates:
    """How fast each axis may move, in units per second."""

    valence: float
    arousal: float
    dominance: float


# Roughly what a face can do. Valence leads because the mouth and brow carry it
# and they are quick; dominance trails because it lives in the neck and
# shoulders, which are heavy. Amusement to gravity is a valence swing of 0.85
# and therefore takes about 700ms to arrive, which is the number this whole
# design is arranged around.
RATES = Rates(valence=1.2, arousal=0.8, dominance=0.6)

# The one exception, and the only reason it exists. When the deterministic
# crisis check fires - see safety/crisis.py - the turn is short-circuited and a
# fixed message is spoken, and the face must already be serious when that
# sentence starts rather than easing into it over most of a second.
CRISIS_REACH_S = 0.25

# Fast enough to land well inside that budget from any starting feeling, and
# deliberately not instant. A face that cuts to an expression in one frame
# reads as a screen changing state, and the moment somebody says they want to
# die is the worst possible moment for the recreation to stop looking human.
CRISIS_RATES = Rates(valence=6.0, arousal=4.0, dominance=3.0)


def _towards(current: float, target: float, limit: float) -> float:
    step = target - current
    if step > limit:
        return current + limit
    if step < -limit:
        return current - limit
    return target


def blend(
    previous: Affect, target: Affect, dt: float, *, rates: Rates = RATES
) -> Affect:
    """Move previous towards target by at most one step's worth of movement.

    The rate table is a parameter rather than a boolean, so the crisis path
    reads as blend(..., rates=CRISIS_RATES) at the call site and there is no
    argument anywhere in the system called something like urgent=True. Passing
    a table names which limits are in force; a flag only names that they are
    not the usual ones.
    """
    if dt <= 0.0 or not math.isfinite(dt):
        return previous

    return Affect(
        valence=_towards(previous.valence, target.valence, rates.valence * dt),
        arousal=_towards(previous.arousal, target.arousal, rates.arousal * dt),
        dominance=_towards(previous.dominance, target.dominance, rates.dominance * dt),
        # Provenance is the destination's from the first frame. The face is not
        # there yet and the coordinates say so; the label is what it is on its
        # way to being, which is what a director scheduling the next gesture
        # needs to know.
        label=target.label,
        intensity=target.intensity,
        confidence=target.confidence,
        source=target.source,
    )
