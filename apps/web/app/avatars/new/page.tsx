"use client";

// Describing the person.
//
// Two fields are absent on purpose. The synthetic-media disclosure is
// generated from the name so it cannot be softened or removed, and the crisis
// line is chosen by country from a list the operator has verified rather than
// typed — a wrong number there is worse than no guardrail at all.

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
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
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [list, langs] = await Promise.all([
          api.countries().then((r) => r.countries),
          api.languages().then((r) => r.languages),
        ]);
        setCountries(list);
        setLanguages(langs);
        if (list.length) {
          setForm((f) => ({
            ...f,
            country: list[0].code,
            locale: list[0].locale,
          }));
          setUnits(list[0].code === "US" ? "ftin" : "cm");
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

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = await api.createAvatar(form);
      router.push(`/upload?avatar=${created.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Could not create the avatar");
    } finally {
      setBusy(false);
    }
  }

  // Split so the height boxes can be narrow. Adding a width alongside w-full
  // does not make them narrow; whichever utility Tailwind emits last wins.
  const box =
    "rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 outline-none focus:border-white/30";
  const field = `w-full ${box}`;

  return (
    <main className="min-h-dvh bg-neutral-950 px-6 py-10 text-neutral-100">
      <form onSubmit={submit} className="mx-auto max-w-xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Who are we recreating?</h1>
          <p className="mt-1.5 text-sm text-neutral-400">
            Everything here is yours to write. It is what the avatar knows about itself.
          </p>
        </div>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">Their name</span>
          <input
            required
            value={form.display_name}
            onChange={(e) => set("display_name", e.target.value)}
            placeholder="How they were known"
            className={field}
          />
        </label>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">Who they were</span>
          <textarea
            required
            rows={4}
            value={form.biography}
            onChange={(e) => set("biography", e.target.value)}
            placeholder="Where they were from, what they did, what mattered to them."
            className={field}
          />
          <span className="block text-xs text-neutral-500">
            Without this the recreation is invented rather than remembered.
          </span>
        </label>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">How they spoke</span>
          <textarea
            rows={3}
            value={form.voice_description}
            onChange={(e) => set("voice_description", e.target.value)}
            placeholder="Blunt and funny. Long pauses. Called everyone 'love'."
            className={field}
          />
        </label>

        {/* Photographs of a face carry no body with them, so this is asked
            rather than worked out. Someone who knew the person is a better
            source than anything a head-and-shoulders picture could support,
            and it keeps how they looked in the family's hands. */}
        <fieldset className="space-y-4 rounded-xl border border-white/10 bg-white/5 p-5">
          <legend className="px-1 text-sm text-neutral-400">How they were built</legend>

          <p className="text-sm text-neutral-400">
            Photographs of someone almost always stop at the shoulders, so the rest of them is the
            part a picture cannot tell us. A few words from you get it close.
          </p>

          <div className="space-y-1.5">
            <span className="block text-sm text-neutral-400">Roughly how tall</span>
            <div className="flex flex-wrap items-center gap-2">
              {units === "cm" ? (
                <input
                  type="number"
                  inputMode="numeric"
                  aria-label="Height in centimetres"
                  value={form.height_cm ?? ""}
                  onChange={(e) => setCentimetres(e.target.value)}
                  placeholder="170"
                  className={`${box} w-28`}
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
                    className={`${box} w-20`}
                  />
                  <span className="text-sm text-neutral-500">ft</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    aria-label="Height in inches"
                    value={inches}
                    onChange={(e) => setImperial(feet, e.target.value)}
                    placeholder="8"
                    className={`${box} w-20`}
                  />
                  <span className="text-sm text-neutral-500">in</span>
                </>
              )}
              <select
                aria-label="Height units"
                value={units}
                onChange={(e) => switchUnits(e.target.value as "cm" | "ftin")}
                className={`${box} w-44`}
              >
                <option value="ftin" className="bg-neutral-900">
                  Feet &amp; inches
                </option>
                <option value="cm" className="bg-neutral-900">
                  Centimetres
                </option>
              </select>
            </div>
            {form.height_cm !== null && !heightLooksWrong && (
              <span className="block text-xs text-neutral-500">{bothUnits(form.height_cm)}</span>
            )}
            {heightLooksWrong && (
              <span className="block text-xs text-amber-200/90">
                That does not look like a height. Have another look at the number.
              </span>
            )}
          </div>

          {/* Words, chosen from a list. There is no right number for any of
              this, and a family should never be asked to invent one. */}
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="block space-y-1.5">
              <span className="text-sm text-neutral-400">Build</span>
              <select
                value={form.build ?? ""}
                onChange={(e) => set("build", (e.target.value || null) as Build | null)}
                className={field}
              >
                <option value="" className="bg-neutral-900">
                  Not sure
                </option>
                {BUILDS.map((option) => (
                  <option key={option.value} value={option.value} className="bg-neutral-900">
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1.5">
              <span className="text-sm text-neutral-400">Shoulders</span>
              <select
                value={form.shoulders ?? ""}
                onChange={(e) => set("shoulders", (e.target.value || null) as ShoulderWidth | null)}
                className={field}
              >
                <option value="" className="bg-neutral-900">
                  Not sure
                </option>
                {SHOULDERS.map((option) => (
                  <option key={option.value} value={option.value} className="bg-neutral-900">
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block space-y-1.5">
              <span className="text-sm text-neutral-400">How they stood</span>
              <select
                value={form.posture ?? ""}
                onChange={(e) => set("posture", (e.target.value || null) as Posture | null)}
                className={field}
              >
                <option value="" className="bg-neutral-900">
                  Not sure
                </option>
                {POSTURES.map((option) => (
                  <option key={option.value} value={option.value} className="bg-neutral-900">
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <p className="text-xs text-neutral-500">
            Leave anything you are not sure about. None of it is needed to carry on, and a blank
            stays blank rather than being filled in for you.
          </p>
        </fieldset>

        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block space-y-1.5">
            <span className="text-sm text-neutral-400">Country</span>
            <select
              required
              value={form.country}
              onChange={(e) => {
                const chosen = countries.find((c) => c.code === e.target.value);
                set("country", e.target.value);
                if (chosen) set("locale", chosen.locale);
              }}
              className={field}
            >
              {countries.map((c) => (
                <option key={c.code} value={c.code} className="bg-neutral-900">
                  {c.name}
                </option>
              ))}
            </select>
            <span className="block text-xs text-neutral-500">
              {countries.find((c) => c.code === form.country)?.crisis_line ??
                "No countries are available yet."}
            </span>
          </label>

          <label className="block space-y-1.5">
            <span className="text-sm text-neutral-400">Language</span>
            {/* A list, not a text box. Typing "SPANISH" here produced an
                avatar that fell back to English prompts while a Spanish voice
                read them aloud, which was unintelligible. */}
            <select
              required
              value={form.locale}
              onChange={(e) => set("locale", e.target.value)}
              className={field}
            >
              {languages.map((l) => (
                <option key={l.code} value={l.code} className="bg-neutral-900">
                  {l.name}
                </option>
              ))}
            </select>
            <span className="block text-xs text-neutral-500">
              What the avatar speaks, and the voice it speaks with.
            </span>
          </label>
        </div>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">Limits (optional)</span>
          <textarea
            rows={2}
            value={form.boundaries}
            onChange={(e) => set("boundaries", e.target.value)}
            placeholder="Anything it should decline to discuss or claim."
            className={field}
          />
          <span className="block text-xs text-neutral-500">
            Left empty, it will refuse to claim it is really them or that it is alive.
          </span>
        </label>

        {countries.length === 0 && (
          <p className="rounded-lg border border-amber-500/25 bg-amber-500/5 px-3 py-2 text-sm text-amber-200/90">
            No countries are available yet. A verified crisis line is required before an avatar can
            be created.
          </p>
        )}

        {error && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy || countries.length === 0 || heightLooksWrong}
          className="w-full rounded-full bg-white px-6 py-3 font-medium text-neutral-950
                     transition hover:bg-neutral-200 disabled:opacity-40"
        >
          {busy ? "Creating…" : "Continue to photographs"}
        </button>
      </form>
    </main>
  );
}
