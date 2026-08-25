"use client";

// A call with one of the customer's avatars.
//
// The disclosure text comes from the server, generated from the avatar's name.
// The client never composes it, so there is no version of this page that can
// display a weaker one.

import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CallStage } from "@/components/CallStage";
import { Disclosure } from "@/components/Disclosure";
import {
  api,
  ApiError,
  ConsentRefused,
  openSession,
  type Avatar,
  type SessionDetails,
} from "@/lib/gateway";

export default function Call() {
  const router = useRouter();
  const params = useParams<{ avatarId: string }>();
  const avatarId = params.avatarId;

  const [avatar, setAvatar] = useState<Avatar | null>(null);
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setAvatar(await api.readAvatar(avatarId));
      } catch (e: unknown) {
        if (e instanceof ApiError && e.status === 401) {
          router.replace("/signin");
          return;
        }
        setError(e instanceof Error ? e.message : "Could not load this avatar");
      }
    })();
  }, [avatarId, router]);

  async function start() {
    setConnecting(true);
    setError(null);
    try {
      setSession(await openSession(avatarId));
    } catch (e: unknown) {
      setError(
        e instanceof ConsentRefused
          ? `This avatar cannot be called yet: ${e.message}`
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
      {/* Outside the call surface, so it stays visible for the whole session
          rather than only before connecting. */}
      <Disclosure text={avatar?.disclosure ?? "You are speaking with a synthetic recreation."} />

      <div className="relative flex-1">
        {session ? (
          <CallStage session={session} onLeave={() => setSession(null)} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-6 px-6">
            <div className="text-center">
              <h1 className="text-3xl font-semibold tracking-tight">
                {avatar?.display_name ?? "Loading…"}
              </h1>
              <p className="mt-2 max-w-md text-sm text-neutral-400">
                Speak normally — you can interrupt at any time, and the camera is part of
                the conversation.
              </p>
            </div>

            <button
              onClick={start}
              disabled={connecting || !avatar}
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
