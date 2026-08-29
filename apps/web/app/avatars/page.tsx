"use client";

// The account's avatars. Empty until the customer creates one — nothing ships
// with a character in it.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { summarise } from "@/lib/body";
import { api, ApiError, type Avatar } from "@/lib/gateway";

export default function Avatars() {
  const router = useRouter();
  const [avatars, setAvatars] = useState<Avatar[] | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setAvatars((await api.listAvatars()).avatars);
      } catch (e: unknown) {
        if (e instanceof ApiError && e.status === 401) router.replace("/signin");
      }
    })();
  }, [router]);

  return (
    <main className="min-h-dvh bg-neutral-950 px-6 py-10 text-neutral-100">
      <div className="mx-auto max-w-3xl space-y-8">
        <header className="flex items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Your avatars</h1>
            <p className="mt-1.5 text-sm text-neutral-400">
              Each one is a person you describe and provide photographs of.
            </p>
          </div>
          <Link
            href="/avatars/new"
            className="shrink-0 rounded-full bg-white px-5 py-2.5 text-sm font-medium
                       text-neutral-950 transition hover:bg-neutral-200"
          >
            New avatar
          </Link>
        </header>

        {avatars === null && <p className="text-sm text-neutral-500">Loading…</p>}

        {avatars?.length === 0 && (
          <div className="rounded-xl border border-white/10 bg-white/5 p-8 text-center">
            <p className="text-neutral-300">You have not created an avatar yet.</p>
            <p className="mx-auto mt-2 max-w-md text-sm text-neutral-500">
              You will need a name, a short description of who they were, and 20–30
              photographs.
            </p>
          </div>
        )}

        <ul className="space-y-3">
          {avatars?.map((avatar) => {
            const body = summarise(avatar.body.stated);
            return (
              <li key={avatar.id} className="rounded-xl border border-white/10 bg-white/5 p-5">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div className="min-w-0">
                    <h2 className="font-medium">{avatar.display_name}</h2>
                    <p className="mt-1 line-clamp-2 text-sm text-neutral-400">{avatar.biography}</p>
                    {/* What the family said about the body, or plainly that
                      they did not. Neither reads as a thing left undone. */}
                    <p className="mt-2 text-xs text-neutral-500">
                      {body ?? "Body not described — a neutral build is used."}
                    </p>
                    <p className="mt-1 text-xs text-neutral-500">
                      {avatar.crisis_line
                        ? `Crisis line: ${avatar.crisis_line.name} (${avatar.crisis_line.number})`
                        : "Not usable yet"}
                    </p>
                  </div>

                  <div className="flex shrink-0 gap-2">
                    <Link
                      href={`/upload?avatar=${avatar.id}`}
                      className="rounded-full bg-white/10 px-4 py-2 text-sm hover:bg-white/20"
                    >
                      {avatar.photo_set_id ? "Photographs" : "Add photographs"}
                    </Link>
                    <Link
                      href={`/call/${avatar.id}`}
                      className={`rounded-full px-4 py-2 text-sm ${
                        avatar.callable
                          ? "bg-white font-medium text-neutral-950 hover:bg-neutral-200"
                          : "pointer-events-none bg-white/5 text-neutral-600"
                      }`}
                    >
                      Call
                    </Link>
                  </div>
                </div>

                {!avatar.callable && (
                  <p className="mt-3 text-xs text-neutral-500">
                    {avatar.has_assets
                      ? "Waiting on verified consent before this can be called."
                      : "Add photographs and build the avatar before calling."}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </main>
  );
}
