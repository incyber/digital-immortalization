"""Crisis lines by country.

The guardrail speaks a real number to somebody in distress. A wrong or
placeholder number there is worse than having no guardrail at all, because it
consumes the one moment when the person was reaching out.

So this is a fixed registry rather than a free-text field on the avatar form.
Letting customers type a number means somebody eventually types their own, or
a placeholder, or nothing.

Verification is an explicit act by whoever runs this service, not a claim made
by whoever wrote the table. Every entry ships unverified; a country becomes
selectable only when its code appears in CRISIS_LINES_VERIFIED, which is how
the operator records that they have checked the number against the operator's
own published source. Numbers change, so this needs re-checking periodically.

The consequence is deliberate: with nothing attested, no avatar can be
created, and the error says exactly why. That is the correct failure for a
product whose safety message is a phone number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrisisLine:
    country: str          # ISO 3166-1 alpha-2
    country_name: str
    locale: str           # default language for this country
    name: str
    number: str
    def verified_by(self, attested: frozenset[str]) -> bool:
        return self.country in attested


# Kept deliberately short. A country absent here cannot be served yet, which is
# the honest position: the product should not operate somewhere it cannot point
# a distressed person at real help.
CRISIS_LINES: tuple[CrisisLine, ...] = (
    CrisisLine("US", "United States", "en", "988 Suicide & Crisis Lifeline", "988"),
    CrisisLine("CA", "Canada", "en", "9-8-8 Suicide Crisis Helpline", "988"),
    CrisisLine("GB", "United Kingdom", "en", "Samaritans", "116 123"),
    CrisisLine("IE", "Ireland", "en", "Samaritans", "116 123"),
    CrisisLine("ES", "Spain", "es", "Línea de Atención a la Conducta Suicida", "024"),
    CrisisLine("MX", "Mexico", "es", "Línea de la Vida", "800 911 2000"),
    CrisisLine("AR", "Argentina", "es", "Línea 135", "135"),
    CrisisLine("CO", "Colombia", "es", "Línea 106", "106"),
    CrisisLine("CL", "Chile", "es", "Salud Responde", "600 360 7777"),
    CrisisLine("AU", "Australia", "en", "Lifeline", "13 11 14"),
    CrisisLine("NZ", "New Zealand", "en", "1737 Need to talk?", "1737"),
    CrisisLine("FR", "France", "fr", "3114", "3114"),
    CrisisLine("DE", "Germany", "de", "Telefonseelsorge", "0800 111 0111"),
    CrisisLine("IT", "Italy", "it", "Telefono Amico", "02 2327 2327"),
    CrisisLine("BR", "Brazil", "pt", "CVV", "188"),
    CrisisLine("PT", "Portugal", "pt", "SNS 24", "808 24 24 24"),
)

BY_COUNTRY = {line.country: line for line in CRISIS_LINES}


class UnsupportedCountry(ValueError):
    """Raised when no verified crisis line exists for a country.

    Deliberately fatal at avatar creation. An avatar with no crisis line is an
    avatar that cannot safely be spoken to.
    """


def parse_attested(raw: str) -> frozenset[str]:
    """Country codes the operator has attested, from configuration."""
    return frozenset(part.strip().upper() for part in (raw or "").split(",") if part.strip())


def for_country(country: str, attested: frozenset[str]) -> CrisisLine:
    line = BY_COUNTRY.get((country or "").upper())
    if line is None:
        raise UnsupportedCountry(
            f"no crisis line on file for {country!r}; this product cannot be "
            "offered in a country where it cannot direct someone to real help"
        )
    if not line.verified_by(attested):
        raise UnsupportedCountry(
            f"the crisis line for {line.country_name} ({line.name}, {line.number}) "
            "has not been verified. Check it against the operator's published "
            f"number, then add {line.country} to CRISIS_LINES_VERIFIED."
        )
    return line


def selectable(attested: frozenset[str]) -> list[CrisisLine]:
    """Countries an avatar may currently be created for."""
    return [line for line in CRISIS_LINES if line.verified_by(attested)]
