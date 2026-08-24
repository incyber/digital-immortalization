"""The guardrail runs before the model, so breaking character is not a decision
the model gets to make. These tests pin the mechanism, including its known and
deliberate limits."""
from avatar.safety.crisis import check


def test_matches_direct_statement_en():
    assert check("i want to kill myself").term == "kill myself"


def test_matches_spanish():
    assert check("quiero matarme", locale="es").term == "matarme"


def test_word_boundaries_not_substrings():
    # A term must match as a whole word. Substring matching would fire here.
    assert check("hablamos de un asesinato en la novela", locale="es") is None


def test_ordinary_speech_does_not_match():
    # "kill" alone is not a term; only the full phrase is. A keyword list
    # cannot do better than this, which is why the design calls it a mechanism
    # demonstration rather than a classifier.
    assert check("i could kill for a coffee") is None


def test_case_and_accent_insensitive():
    assert check("QUIERO MATARME", locale="es") is not None
    assert check("quiero suicidarme", locale="es") is not None


def test_unknown_locale_falls_back_to_english():
    assert check("i want to kill myself", locale="fr") is not None


def test_empty_input_is_safe():
    assert check("") is None
