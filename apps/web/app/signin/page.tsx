"use client";

// Sign in or create an account. One form for both, because at this stage of
// the product the distinction is not worth a second page.

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/gateway";

export default function SignIn() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("register");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? api.login(email, password) : api.register(email, password));
      router.push("/upload");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="flex min-h-dvh items-center justify-center bg-neutral-950 px-6 text-neutral-100">
      <form onSubmit={submit} className="w-full max-w-sm space-y-5">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {mode === "login" ? "Sign in" : "Create an account"}
          </h1>
          <p className="mt-1.5 text-sm text-neutral-400">
            Your photographs are stored privately and are never shared between accounts.
          </p>
        </div>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">Email</span>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5
                       outline-none focus:border-white/30"
          />
        </label>

        <label className="block space-y-1.5">
          <span className="text-sm text-neutral-400">Password</span>
          <input
            type="password"
            required
            minLength={12}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2.5
                       outline-none focus:border-white/30"
          />
          <span className="block text-xs text-neutral-500">At least 12 characters.</span>
        </label>

        {error && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-full bg-white px-6 py-2.5 font-medium text-neutral-950
                     transition hover:bg-neutral-200 disabled:opacity-50"
        >
          {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
        </button>

        <button
          type="button"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
          className="w-full text-sm text-neutral-400 underline-offset-4 hover:underline"
        >
          {mode === "login" ? "Create an account instead" : "I already have an account"}
        </button>
      </form>
    </main>
  );
}
