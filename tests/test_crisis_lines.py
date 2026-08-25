import pytest

from avatar.safety.crisis_lines import (
    CRISIS_LINES,
    UnsupportedCountry,
    for_country,
    parse_attested,
    selectable,
)

NONE = frozenset()
SOME = frozenset({"US", "ES"})


def test_nothing_is_selectable_until_attested():
    # The default state of the product: no country served, because nobody has
    # confirmed any number yet.
    assert selectable(NONE) == []


def test_an_unattested_country_is_refused_with_instructions():
    with pytest.raises(UnsupportedCountry, match="CRISIS_LINES_VERIFIED"):
        for_country("US", NONE)


def test_an_attested_country_resolves():
    line = for_country("US", SOME)
    assert line.number == "988"


def test_an_unknown_country_is_refused(): 
    with pytest.raises(UnsupportedCountry, match="no crisis line on file"):
        for_country("ZZ", SOME)


def test_an_empty_country_is_refused():
    with pytest.raises(UnsupportedCountry):
        for_country("", SOME)


def test_attesting_one_country_does_not_attest_another():
    for_country("ES", SOME)
    with pytest.raises(UnsupportedCountry):
        for_country("MX", SOME)


def test_parse_attested_handles_spacing_and_case():
    assert parse_attested(" us , es ,mx") == frozenset({"US", "ES", "MX"})
    assert parse_attested("") == frozenset()


def test_no_entry_has_a_placeholder_number():
    for line in CRISIS_LINES:
        assert line.number.strip()
        assert not any(bad in line.number.upper() for bad in ("TBD", "TODO", "XXX", "N/A"))
        assert line.name.strip()


def test_country_codes_are_unique_and_well_formed():
    codes = [line.country for line in CRISIS_LINES]
    assert len(codes) == len(set(codes))
    assert all(len(c) == 2 and c.isupper() for c in codes)
