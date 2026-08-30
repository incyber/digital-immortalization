"use client";

// Sign in or create an account. One form for both, because at this stage of
// the product the distinction is not worth a second page.
//
// Two fields, and the browser fills them: the autocomplete tokens are set
// correctly so a saved password arrives on its own and nobody is asked to
// remember anything on the worst week of their life.

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Field } from "@/components/ui/Field";
import { Notice } from "@/components/ui/Notice";
import { Screen } from "@/components/ui/Screen";
import { useSharedDemo } from "@/lib/demo";
import { api } from "@/lib/gateway";

export default function SignIn() {
  const router = useRouter();

  // A shared demo has one account and it takes no credentials, so this form
  // has nothing to submit to — the gateway refuses both endpoints. Nothing
  // links here in that mode, because nothing returns a 401 to redirect on;
  // this is for somebody who typed the address or followed an old link, and
  // sending them to the entry point is better than a form that only fails.
  const shared = useSharedDemo();
  useEffect(() => {
    if (shared) router.replace("/");
  }, [shared, router]);

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
      // Back to the entry point, which knows whether there is anybody to call
      // yet. It is the only place that decision is made.
      router.push("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen
      title={mode === "login" ? "Welcome back." : "Start with an account."}
      lede="Your photographs are stored privately. They are never shared between accounts, and you can remove them at any time."
      measure="tight"
      center
    >
      <form onSubmit={submit} className="space-y-6">
        <Field label="Email">
          <input
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="entry"
          />
        </Field>

        <Field
          label="Password"
          help={mode === "register" ? "At least twelve characters." : undefined}
        >
          <input
            type="password"
            required
            minLength={12}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="entry"
          />
        </Field>

        {error && (
          <Notice tone="problem" role="alert">
            {error}
          </Notice>
        )}

        <div className="space-y-4 pt-2">
          <Button type="submit" rank="filled" wide disabled={busy}>
            {busy ? "One moment…" : mode === "login" ? "Sign in" : "Continue"}
          </Button>

          <div className="text-center">
            <Button
              rank="plain"
              small
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError(null);
              }}
            >
              {mode === "login" ? "Create an account instead" : "I already have an account"}
            </Button>
          </div>
        </div>
      </form>
    </Screen>
  );
}
