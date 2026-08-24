"""Deterministic crisis check.

Runs ahead of the language model in the pipeline. A match short-circuits the
turn: the model is never called, a fixed message is spoken, and the event is
written to safety_events. Placing this outside the model is the entire point -
a model asked to break character may decline to.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from avatar.safety.keywords import DEFAULT_LOCALE, SAFETY_TEMPLATE, TERMS


@dataclass(frozen=True)
class CrisisMatch:
    """The term that fired and the locale whose list it came from."""

    term: str
    locale: str


def normalise(text: str) -> str:
    """Lowercase and strip accents, so 'QUIERO MATARME' and 'quiero matarme'
    and any accented spelling all reach the term list in the same shape."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=None)
def _patterns(locale: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Compiled whole-word patterns for a locale, built once per process.

    \\b anchors are what keep 'asesinato' from matching a term contained inside
    it; substring matching would produce false positives on ordinary speech.
    """
    terms = TERMS.get(locale) or TERMS[DEFAULT_LOCALE]
    return tuple((t, re.compile(rf"\b{re.escape(t)}\b")) for t in terms)


def check(text: str, locale: str = DEFAULT_LOCALE) -> CrisisMatch | None:
    """Return the first matching term, or None.

    An unknown locale falls back to English rather than failing open with no
    checking at all.
    """
    if not text or not text.strip():
        return None

    resolved = locale if locale in TERMS else DEFAULT_LOCALE
    haystack = normalise(text)
    for term, pattern in _patterns(resolved):
        if pattern.search(haystack):
            return CrisisMatch(term=term, locale=resolved)
    return None


def safety_reply(locale: str, line_name: str, line_number: str) -> str:
    """The fixed message spoken in place of a model reply."""
    template = SAFETY_TEMPLATE.get(locale) or SAFETY_TEMPLATE[DEFAULT_LOCALE]
    return template.format(line_name=line_name, line_number=line_number)
