"use client";

// A call with one of the customer's avatars.
//
// The disclosure text comes from the server, generated from the avatar's name.
// The client never composes it, so there is no version of this page that can
// display a weaker one.
//
// The whole screen is dark in both appearances. A face on video belongs on
// black, and this is not the moment to be adjusting to a white interface.

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CallStage } from "@/components/CallStage";
import { Disclosure } from "@/components/Disclosure";
import { Button } from "@/components/ui/Button";
import { Notice } from "@/components/ui/Notice";
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
  const [refused, setRefused] = useState<string | null>(null);
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
    setRefused(null);
    try {
      setSession(await openSession(avatarId));
    } catch (e: unknown) {
      // A refusal is not a failure. It means the consent record is not yet
      // verified, which is a thing somebody is waiting on rather than a thing
      // they did wrong, and it is said in those terms.
      if (e instanceof ConsentRefused) setRefused(e.message);
      else setError(e instanceof Error ? e.message : "Could not start the call");
    } finally {
      setConnecting(false);
    }
  }

  return (
    <main className="always-dark flex h-dvh flex-col bg-surface text-label">
      {/* Outside the call surface, so it stays visible for the whole session
          rather than only before connecting. */}
      <Disclosure text={avatar?.disclosure ?? "You are speaking with a synthetic recreation."} />

      <div className="relative flex-1">
        {session ? (
          <CallStage session={session} onLeave={() => setSession(null)} />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-8 px-6 py-10">
            <div className="max-w-md text-center">
              <h1 className="text-large-title text-label">
                {avatar?.display_name ?? "One moment."}
              </h1>
              <p className="mt-4 text-body text-label-secondary">
                Speak normally. You can interrupt at any time, and take as long as you like.
              </p>
            </div>

            <Button rank="filled" onClick={start} disabled={connecting || !avatar}>
              {connecting ? "Connecting…" : "Start the call"}
            </Button>

            {refused && (
              <Notice
                tone="attention"
                role="status"
                title="Not ready for a call yet."
                className="max-w-md text-left"
              >
                {/* The reason is the server's own words, shown as it wrote
                    them. Nothing here rephrases a legal state into something
                    friendlier than it is. */}
                <p className="first-letter:uppercase">{refused}.</p>
                <p className="mt-2 text-footnote text-label-tertiary">
                  Permission has to be verified before any call can open.
                </p>
              </Notice>
            )}

            {error && (
              <Notice tone="problem" role="alert" className="max-w-md text-left">
                {error}
              </Notice>
            )}

            {/* Quietly present for the whole of the pre-call screen. Somebody
                who needs this number should not have to go looking. */}
            {avatar?.crisis_line && (
              <p className="text-footnote text-label-secondary">
                If you need someone to talk to · {avatar.crisis_line.name}{" "}
                {avatar.crisis_line.number}
              </p>
            )}

            <Link
              href="/avatars"
              className="text-subhead text-accent underline-offset-4 hover:underline"
            >
              Not now
            </Link>
          </div>
        )}
      </div>
    </main>
  );
}
