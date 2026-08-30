"use client";

// A call with one of the customer's avatars.
//
// The disclosure text comes from the server, generated from the avatar's name.
// The client never composes it, so there is no version of this page that can
// display a weaker one. The same is true of the second line: when a likeness
// has been built, how much of it was generated rather than photographed is
// the build's own sentence, passed through untouched.
//
// Which renderer draws the person is settled here, before any connection is
// opened, and handed to the call surface already decided. Deciding late would
// mean deciding during the call, and a picture that changed character halfway
// through would be the loudest thing on the screen.
//
// The whole screen is dark in both appearances. A face on video belongs on
// black, and this is not the moment to be adjusting to a white interface.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { CallStage } from "@/components/CallStage";
import { Disclosure } from "@/components/Disclosure";
import { Button } from "@/components/ui/Button";
import { Notice } from "@/components/ui/Notice";
import {
  api,
  ApiError,
  ConsentRefused,
  NoLikeness,
  openSession,
  type Avatar,
  type SessionDetails,
} from "@/lib/gateway";
import { resolveLikeness, type LikenessPlan } from "@/lib/likeness";

/** The avatar id in the query string, or null — including during prerender. */
function avatarFromLocation(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("avatar");
}

export default function Call() {
  const router = useRouter();

  // Who is being called arrives in the query string rather than in the path.
  //
  // The site is a static export, and an export writes one file per route at
  // build time. A path segment whose values are avatar ids cannot be
  // enumerated then — there is no list, and there will not be one until a
  // family creates one — so /call/<id> has no file behind it and a deep link
  // is a 404. A query string is invisible to routing: one file answers every
  // avatar.
  //
  // Read off window.location rather than through useSearchParams, so this page
  // needs no suspense boundary — the same choice app/upload/page.tsx makes.
  // Held in no state at all: nothing this component draws depends on it, only
  // the requests it makes, so reading it during render costs nothing and
  // cannot disagree with the prerendered markup.
  const avatarId = avatarFromLocation();

  const [avatar, setAvatar] = useState<Avatar | null>(null);
  const [likeness, setLikeness] = useState<LikenessPlan | null>(null);
  const [session, setSession] = useState<SessionDetails | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  // Held apart from `refused` because the remedy is different: consent is
  // somebody else's signature, a likeness is an upload away.
  const [unbuilt, setUnbuilt] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  useEffect(() => {
    // A call naming nobody. Not an error worth a screen of its own — the list
    // is where they were going.
    if (!avatarId) {
      router.replace("/avatars");
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const loaded = await api.readAvatar(avatarId);
        if (!cancelled) setAvatar(loaded);
      } catch (e: unknown) {
        if (e instanceof ApiError && e.status === 401) {
          router.replace("/signin");
          return;
        }
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load this avatar");
      }
    })();

    // Runs alongside rather than after: it is a second small request, and
    // nobody should wait longer to reach the button because of it. It never
    // rejects — every reason a splat cannot be used comes back as the video
    // track — so there is no failure branch to write here.
    resolveLikeness(avatarId).then((plan) => {
      if (!cancelled) setLikeness(plan);
    });

    return () => {
      cancelled = true;
    };
  }, [avatarId, router]);

  async function start() {
    if (!avatarId) return;
    setConnecting(true);
    setError(null);
    setRefused(null);
    setUnbuilt(null);
    try {
      setSession(await openSession(avatarId));
    } catch (e: unknown) {
      // A refusal is not a failure. It means the consent record is not yet
      // verified, which is a thing somebody is waiting on rather than a thing
      // they did wrong, and it is said in those terms.
      if (e instanceof ConsentRefused) setRefused(e.message);
      else if (e instanceof NoLikeness) setUnbuilt(e.message);
      else setError(e instanceof Error ? e.message : "Could not start the call");
    } finally {
      setConnecting(false);
    }
  }

  return (
    <main className="always-dark flex h-dvh flex-col bg-surface text-label">
      {/* Outside the call surface, so it stays visible for the whole session
          rather than only before connecting. */}
      <Disclosure
        text={avatar?.disclosure ?? "You are speaking with a synthetic recreation."}
        detail={likeness?.splat?.disclosure}
      />

      <div className="relative flex-1">
        {session && likeness ? (
          <CallStage session={session} likeness={likeness} onLeave={() => setSession(null)} />
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

            {/* The likeness decision gates the button as much as the avatar
                does: starting before it has landed would connect the room and
                only then discover what should be drawn in it. */}
            <Button rank="filled" onClick={start} disabled={connecting || !avatar || !likeness}>
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

            {unbuilt && (
              <Notice
                tone="attention"
                role="status"
                title="There is no likeness yet."
                className="max-w-md text-left"
              >
                <p className="first-letter:uppercase">{unbuilt}.</p>
                <p className="mt-2 text-footnote text-label-tertiary">
                  A call only ever shows the real person, so it waits until one
                  has been built.
                </p>
                <Link
                  href={`/avatars/${avatarId}`}
                  className="mt-3 inline-block text-footnote underline"
                >
                  Add a video or photographs
                </Link>
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
