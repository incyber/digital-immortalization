"use client";

// Entry screen. One avatar for now; the roster arrives with the platform
// sub-project.

import { useState } from "react";
import { CallStage } from "@/components/CallStage";
import { Disclosure } from "@/components/Disclosure";
import { ConsentRefused, openSession, type SessionDetails } from "@/lib/api";

const AVATAR_ID = process.env.NEXT_PUBLIC_AVATAR_ID ?? "colon";

const DISCLOSURE =
  "You are speaking with a synthetic recreation built from the historical " +
  "record. It is not the person, and it can be wrong.";

export default function Home() {
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  async function start() {
    setConnecting(true);
    setError(null);
    try {
      setSession(await openSession(AVATAR_ID));
    } catch (e) {
      setError(
        e instanceof ConsentRefused
          ? `This avatar cannot be called: ${e.message}`
          : e instanceof Error
            ? e.message
            : "Could not start the call",
      );
    } finally {
      setConnecting(false);
    }
  }

  return (
    <main className="flex h-dvh flex-col bg-neutral-950 text-neutral-100">
      {/* Disclosure is above the call surface and outside it, so it stays
          visible for the whole session rather than only before connecting. */}
      <Disclosure text={DISCLOSURE} />

      <div className="relative flex-1">
        {session ? (
          <CallStage session={session} onLeave={() => setSession(null)} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-6 px-6">
            <div className="text-center">
              <h1 className="text-3xl font-semibold tracking-tight">Cristóbal Colón</h1>
              <p className="mt-2 max-w-md text-sm text-neutral-400">
                A live call. Speak normally — you can interrupt at any time, and
                the camera is part of the conversation.
              </p>
            </div>

            <button
              onClick={start}
              disabled={connecting}
              className="rounded-full bg-white px-8 py-3 font-medium text-neutral-950
                         transition hover:bg-neutral-200 disabled:opacity-50"
            >
              {connecting ? "Connecting…" : "Start call"}
            </button>

            {error && (
              <p className="max-w-md rounded-lg border border-red-500/30 bg-red-500/10
                            px-4 py-3 text-center text-sm text-red-300">
                {error}
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
