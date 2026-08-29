"use client";

// Describing the person, and recording who authorised the recreation.
//
// Three questions are visible: their name, who they were, and who is giving
// permission. Everything else the gateway needs is either prefilled from the
// browser — country, language, the units a height is typed in — or folded away
// under one line somebody can open if they want to. Nothing in the closed part
// is required, and nothing in it is invented on the family's behalf: a blank
// stays blank all the way to the server.
//
// Two fields are absent on purpose. The synthetic-media disclosure is
// generated from the name so it cannot be softened or removed, and the crisis
// line is chosen by country from a list the operator has verified rather than
// typed — a wrong number there is worse than no guardrail at all.

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Field, FieldGroup } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { Screen } from "@/components/ui/Screen";
import { Segmented } from "@/components/ui/Segmented";
import {
  BUILDS,
  MAX_HEIGHT_CM,
  MIN_HEIGHT_CM,
  POSTURES,
  SHOULDERS,
  bothUnits,
  fromFeetInches,
  toFeetInches,
} from "@/lib/body";
import {
  api,
  ApiError,
  type AvatarInput,
  type Build,
  type Country,
  type Posture,
  type ShoulderWidth,
} from "@/lib/gateway";
import { preferredCountry, preferredLocale, preferredUnits } from "@/lib/prefill";

// What the gateway does with each answer. "self" is the only one it can verify
// on its own, and saying so here is the difference between a gate and a
// formality.
type Relationship = "self" | "family" | "friend";

const RELATIONSHIPS: { value: Relationship; label: string }[] = [
  { value: "family", label: "Family" },
  { value: "friend", label: "Close friend" },
  { value: "self", label: "This is me" },
];

export default function NewAvatar() {
  const router = useRouter();
  const [countries, setCountries] = useState<Country[]>([]);
  const [languages, setLanguages] = useState<{ code: string; name: string }[]>([]);
  const [form, setForm] = useState<AvatarInput>({
    display_name: "",
    locale: "en",
    country: "",
    biography: "",
    voice_description: "",
    boundaries: "",
    // Nobody is obliged to answer these. Unanswered is sent as unanswered.
    height_cm: null,
    build: null,
    shoulders: null,
    posture: null,
  });
  // Which units the height boxes are showing. Both are offered because half
  // the people using this think in feet and the other half in centimetres.
  const [units, setUnits] = useState<"cm" | "ftin">("cm");
  const [feet, setFeet] = useState("");
  const [inches, setInches] = useState("");
  const [relationship, setRelationship] = useState<Relationship | null>(null);
  const [rightsHolder, setRightsHolder] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Held so that a consent call which fails does not create a second person
  // when somebody presses the button again.
  const created = useRef<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [list, langs] = await Promise.all([
          api.countries().then((r) => r.countries),
          api.languages().then((r) => r.languages),
        ]);
        setCountries(list);
        setLanguages(langs);

        // Prefilled from the browser rather than asked. Both remain editable
        // under the details below.
        const country = preferredCountry(list);
        if (country) {
          setForm((f) => ({
            ...f,
            country: country.code,
            locale: preferredLocale(langs, country),
          }));
          setUnits(preferredUnits(country));
        }
      } catch (e: unknown) {
        if (e instanceof ApiError && e.status === 401) router.replace("/signin");
      }
    })();
  }, [router]);

  function set<K extends keyof AvatarInput>(key: K, value: AvatarInput[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  // Height is held once, in centimetres, whichever boxes it was typed into.
  // Keeping two editable copies of one number is how they end up disagreeing.
  function setCentimetres(raw: string) {
    const cm = Number(raw);
    set("height_cm", raw.trim() === "" || Number.isNaN(cm) ? null : Math.round(cm));
  }

  function setImperial(nextFeet: string, nextInches: string) {
    setFeet(nextFeet);
    setInches(nextInches);
    const ft = Number(nextFeet);
    const inch = nextInches.trim() === "" ? 0 : Number(nextInches);
    const blank = nextFeet.trim() === "";
    set(
      "height_cm",
      blank || Number.isNaN(ft) || Number.isNaN(inch) ? null : fromFeetInches(ft, inch),
    );
  }

  function switchUnits(next: "cm" | "ftin") {
    setUnits(next);
    if (next === "ftin" && form.height_cm !== null) {
      const imperial = toFeetInches(form.height_cm);
      setFeet(String(imperial.feet));
      setInches(String(imperial.inches));
    }
  }

  // Caught here so somebody who slips a digit is told plainly, rather than
  // having the number quietly moved into range on their behalf.
  const heightLooksWrong =
    form.height_cm !== null && (form.height_cm < MIN_HEIGHT_CM || form.height_cm > MAX_HEIGHT_CM);

  const chosenCountry = countries.find((c) => c.code === form.country) ?? null;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!relationship) return;
    setBusy(true);
    setError(null);
    try {
      // A second press after something went wrong updates the person who was
      // already created rather than making another one, so an edit made
      // between the two attempts is the version that is kept.
      let id = created.current;
      if (id) await api.updateAvatar(id, form);
      else id = (await api.createAvatar(form)).id;
      created.current = id;

      // Recorded in the same action, so there is no screen anywhere in this
      // product where a recreation exists without a named person behind it.
      await api.recordConsent(id, {
        rights_holder_name: rightsHolder.trim(),
        relationship_to_subject: relationship,
        jurisdiction: chosenCountry?.name ?? form.country,
      });

      router.push(`/upload?avatar=${id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not save this");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen
      back={{ href: "/avatars", label: "Back" }}
      title="Tell us who they were."
      lede="Three things to start. Everything else can be left alone, or filled in later."
    >
      <form onSubmit={submit} className="space-y-8">
        <Field label="Their name">
          <input
            required
            value={form.display_name}
            onChange={(e) => set("display_name", e.target.value)}
            placeholder="How they were known"
            className="entry"
          />
        </Field>

        <Field
          label="Who they were"
          help="Where they were from, what they did, what mattered to them. A few sentences is enough — without them the recreation is invented rather than remembered."
        >
          <textarea
            required
            rows={5}
            value={form.biography}
            onChange={(e) => set("biography", e.target.value)}
            placeholder="Born in Cádiz. Taught maths for thirty years. Made everybody laugh at the table."
            className="entry"
          />
        </Field>

        {/* The consent gate. Calm, and unmissable because of where it sits and
            what it asks for — never because of how loudly it is coloured. It
            is not optional, it is not a checkbox, and the name typed here is
            recorded against the recreation. */}
        <section className="border-t border-separator pt-8">
          <h2 className="text-title-3 text-label">Permission</h2>
          <p className="mt-2 text-subhead text-label-secondary">
            Recreating a person needs someone who can speak for them. Who is doing that is kept
            on file with this recreation, and can be withdrawn at any time.
          </p>

          <div className="mt-6 space-y-6">
            <FieldGroup label="Your relationship to them">
              <Segmented
                label="Your relationship to them"
                value={relationship}
                options={RELATIONSHIPS}
                onChange={setRelationship}
              />
            </FieldGroup>

            <Field label="Your full name">
              <input
                required
                autoComplete="name"
                value={rightsHolder}
                onChange={(e) => setRightsHolder(e.target.value)}
                placeholder="The name you would sign"
                className="entry"
              />
            </Field>

            {relationship && (
              <Notice tone="quiet">
                {relationship === "self"
                  ? "You are recreating yourself, so nobody else's permission is engaged. This is recorded as soon as you continue."
                  : "Someone reads this before the first call can open. It usually takes a day. You can add photographs in the meantime."}
              </Notice>
            )}
          </div>
        </section>

        {/* Everything the gateway can manage without asking. Open only by
            somebody who wants to; closed, this is one line of grey text. */}
        <details className="group border-t border-separator pt-8">
          <summary
            className="flex cursor-pointer list-none items-center gap-2 text-body text-accent
                       transition-opacity duration-200 hover:opacity-70
                       [&::-webkit-details-marker]:hidden"
          >
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              aria-hidden="true"
              fill="none"
              className="transition-transform duration-200 group-open:rotate-90"
            >
              <path
                d="M4 2l4 4-4 4"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            More about them
          </summary>

          <div className="mt-6 space-y-8">
            <Field
              label="How they spoke"
              optional
              help="Turns of phrase, pace, the things they always said."
            >
              <textarea
                rows={3}
                value={form.voice_description}
                onChange={(e) => set("voice_description", e.target.value)}
                placeholder="Blunt and funny. Long pauses. Called everyone 'love'."
                className="entry"
              />
            </Field>

            {/* Photographs of a face carry no body with them, so this is asked
                rather than worked out. Someone who knew the person is a better
                source than anything a head-and-shoulders picture could
                support, and it keeps how they looked in the family's hands. */}
            <FieldGroup
              label="How they were built"
              optional
              help="Leave anything you are not sure about. A blank stays blank rather than being filled in for you."
            >
              <div className="space-y-6">
                <div>
                  <span className="mb-2 block text-footnote text-label-secondary">
                    Roughly how tall
                  </span>
                  <div className="flex flex-wrap items-center gap-3">
                    {units === "cm" ? (
                      <input
                        type="number"
                        inputMode="numeric"
                        aria-label="Height in centimetres"
                        value={form.height_cm ?? ""}
                        onChange={(e) => setCentimetres(e.target.value)}
                        placeholder="170"
                        className="entry w-24"
                      />
                    ) : (
                      <>
                        <input
                          type="number"
                          inputMode="numeric"
                          aria-label="Height in feet"
                          value={feet}
                          onChange={(e) => setImperial(e.target.value, inches)}
                          placeholder="5"
                          className="entry w-20"
                        />
                        <span className="text-subhead text-label-tertiary">ft</span>
                        <input
                          type="number"
                          inputMode="numeric"
                          aria-label="Height in inches"
                          value={inches}
                          onChange={(e) => setImperial(feet, e.target.value)}
                          placeholder="8"
                          className="entry w-20"
                        />
                        <span className="text-subhead text-label-tertiary">in</span>
                      </>
                    )}
                    <Segmented
                      label="Height units"
                      value={units}
                      options={[
                        { value: "ftin", label: "ft / in" },
                        { value: "cm", label: "cm" },
                      ]}
                      onChange={switchUnits}
                      className="max-w-[11rem]"
                    />
                  </div>
                  {form.height_cm !== null && !heightLooksWrong && (
                    <span className="mt-2 block text-footnote tabular-nums text-label-secondary">
                      {bothUnits(form.height_cm)}
                    </span>
                  )}
                  {heightLooksWrong && (
                    <span className="mt-2 block text-footnote text-orange">
                      That does not look like a height. Have another look at the number.
                    </span>
                  )}
                </div>

                {/* Words, chosen from a list. There is no right number for any
                    of this, and a family should never be asked to invent one. */}
                <div className="grid gap-6 sm:grid-cols-3">
                  <Field label="Build">
                    <select
                      value={form.build ?? ""}
                      onChange={(e) => set("build", (e.target.value || null) as Build | null)}
                      className="entry"
                    >
                      <option value="">Not sure</option>
                      {BUILDS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Shoulders">
                    <select
                      value={form.shoulders ?? ""}
                      onChange={(e) =>
                        set("shoulders", (e.target.value || null) as ShoulderWidth | null)
                      }
                      className="entry"
                    >
                      <option value="">Not sure</option>
                      {SHOULDERS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="How they stood">
                    <select
                      value={form.posture ?? ""}
                      onChange={(e) => set("posture", (e.target.value || null) as Posture | null)}
                      className="entry"
                    >
                      <option value="">Not sure</option>
                      {POSTURES.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
              </div>
            </FieldGroup>

            <div className="grid gap-6 sm:grid-cols-2">
              <Field
                label="Country"
                help={chosenCountry?.crisis_line ?? "No countries are available yet."}
              >
                <select
                  required
                  value={form.country}
                  onChange={(e) => {
                    const chosen = countries.find((c) => c.code === e.target.value);
                    set("country", e.target.value);
                    if (chosen) set("locale", chosen.locale);
                  }}
                  className="entry"
                >
                  {countries.map((c) => (
                    <option key={c.code} value={c.code}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </Field>

              {/* A list, not a text box. Typing "SPANISH" here produced an
                  avatar that fell back to English prompts while a Spanish
                  voice read them aloud, which was unintelligible. */}
              <Field label="Language" help="What they speak, and the voice they speak with.">
                <select
                  required
                  value={form.locale}
                  onChange={(e) => set("locale", e.target.value)}
                  className="entry"
                >
                  {languages.map((l) => (
                    <option key={l.code} value={l.code}>
                      {l.name}
                    </option>
                  ))}
                </select>
              </Field>
            </div>

            <Field
              label="Limits"
              optional
              help="Left empty, they will refuse to claim they are really them, or that they are alive."
            >
              <textarea
                rows={2}
                value={form.boundaries}
                onChange={(e) => set("boundaries", e.target.value)}
                placeholder="Anything they should decline to discuss or claim."
                className="entry"
              />
            </Field>
          </div>
        </details>

        {countries.length === 0 && (
          <Notice tone="attention" title="Not available here yet.">
            A verified crisis line is required in your country before a recreation can be made.
          </Notice>
        )}

        {error && (
          <Notice tone="problem" role="alert">
            {error}
          </Notice>
        )}

        <div className="border-t border-separator pt-8">
          <Button
            type="submit"
            rank="filled"
            wide
            disabled={busy || countries.length === 0 || heightLooksWrong || !relationship}
          >
            {busy ? "One moment…" : "Continue to photographs"}
          </Button>
          {!relationship && (
            <p className="mt-4 text-center text-footnote text-label-secondary">
              Choose your relationship to them to continue.
            </p>
          )}
        </div>
      </form>
    </Screen>
  );
}
