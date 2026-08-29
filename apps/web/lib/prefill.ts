// What the browser already knows.
//
// Nobody should be asked which country they are in by a machine that is
// holding their timezone and their language settings. These read what is
// there, choose the closest thing the gateway actually supports, and fall
// back quietly when the guess is not on the list. Every one of them is
// changeable afterwards — a prefilled answer is a starting point, not a claim.

import type { Country } from "@/lib/gateway";

/** The browser's region, as a two-letter code, or null if it will not say. */
function region(): string | null {
  if (typeof navigator === "undefined") return null;
  const tags = navigator.languages?.length ? navigator.languages : [navigator.language];
  for (const tag of tags) {
    if (!tag) continue;
    try {
      const found = new Intl.Locale(tag).maximize().region;
      if (found) return found;
    } catch {
      // A tag the browser cannot parse tells us nothing. Try the next one.
    }
  }
  return null;
}

/** The browser's language, as a two-letter code, or null. */
function language(): string | null {
  if (typeof navigator === "undefined") return null;
  const tag = navigator.languages?.[0] ?? navigator.language;
  return tag ? tag.split("-")[0].toLowerCase() : null;
}

/**
 * The country to start on: theirs if the operator supports it, otherwise the
 * first one on the list. Never null when a list was supplied, because a screen
 * with an empty country box asks a question nobody wants to be asked today.
 */
export function preferredCountry(countries: Country[]): Country | null {
  if (countries.length === 0) return null;
  const code = region();
  return countries.find((c) => c.code === code) ?? countries[0];
}

/**
 * The language to start on: theirs if there is a voice for it, otherwise
 * whatever the chosen country speaks.
 */
export function preferredLocale(
  languages: { code: string }[],
  country: Country | null,
): string {
  const code = language();
  if (code && languages.some((l) => l.code === code)) return code;
  if (country && languages.some((l) => l.code === country.locale)) return country.locale;
  return languages[0]?.code ?? "en";
}

/** Feet and inches for the places that use them, centimetres everywhere else. */
export function preferredUnits(country: Country | null): "cm" | "ftin" {
  return country?.code === "US" || country?.code === "GB" ? "ftin" : "cm";
}
