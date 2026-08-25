"use client";

// Describing the person.
//
// Two fields are absent on purpose. The synthetic-media disclosure is
// generated from the name so it cannot be softened or removed, and the crisis
// line is chosen by country from a list the operator has verified rather than
// typed — a wrong number there is worse than no guardrail at all.

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, ApiError, type Country } from "@/lib/gateway";

export default function NewAvatar() {
  const router = useRouter();
  const [countries, setCountries] = useState<Country[]>([]);
  const [form, setForm] = useState({
    display_name: "",
    locale: "en",
    country: "",
    biography: "",
    voice_description: "",
    boundaries: "",
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const list = (await api.countries()).countries;
        setCountries(list);
        if (list.length) {
          setForm((f) => ({ ...f, country: list[0].code, locale: list[0].locale }));
        }
      } catch (e: unknown) {
        if (e instanceof ApiError && e.status === 401) router.replace("/signin");
      }
    })();
  }, [router]);

  function set<K extends keyof typeof form>(key: K, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
  }

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

  const field =
    "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5 outline-none focus:border-white/30";

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
            <input
              required
              value={form.locale}
              onChange={(e) => set("locale", e.target.value)}
              className={field}
            />
            <span className="block text-xs text-neutral-500">
              The language the avatar speaks.
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
            No countries are available yet. A verified crisis line is required before an
            avatar can be created.
          </p>
        )}

        {error && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy || countries.length === 0}
          className="w-full rounded-full bg-white px-6 py-3 font-medium text-neutral-950
                     transition hover:bg-neutral-200 disabled:opacity-40"
        >
          {busy ? "Creating…" : "Continue to photographs"}
        </button>
      </form>
    </main>
  );
}
