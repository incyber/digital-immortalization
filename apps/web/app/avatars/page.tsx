"use client";

// The account's avatars. Empty until the customer creates one — nothing ships
// with a character in it.
//
// A grouped list, in Apple's sense: rows separated by hairlines rather than
// each one drawn as its own card. What is on a row is what somebody would ask
// about it — who they are, whether they can be called, and if not, what is
// still needed.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { controlClass } from "@/components/ui/Button";
import { Notice } from "@/components/ui/Notice";
import { Screen } from "@/components/ui/Screen";
import { summarise } from "@/lib/body";
import { api, ApiError, type Avatar } from "@/lib/gateway";

function Row({ avatar }: { avatar: Avatar }) {
  const body = summarise(avatar.body.stated);

  const state = avatar.callable
    ? null
    : avatar.has_assets
      ? "Waiting on the consent record before this call can open."
      : avatar.photo_set_id
        ? "Photographs added. They still need to be built into a likeness."
        : "Add photographs, and they can be built.";

  return (
    <li className="py-6 first:pt-0 last:pb-0 [&+li]:border-t [&+li]:border-separator">
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="min-w-0 flex-1">
          <h2 className="text-headline text-label">{avatar.display_name}</h2>
          {avatar.biography && (
            <p className="mt-1 line-clamp-2 text-subhead text-label-secondary">
              {avatar.biography}
            </p>
          )}
          {/* What the family said about the body, or plainly that they did
              not. Neither reads as a thing left undone. */}
          <p className="mt-2 text-footnote text-label-secondary">
            {body ?? "Body not described — a neutral build is used."}
          </p>
          <p className="mt-1 text-footnote text-label-secondary">
            {avatar.crisis_line
              ? `Crisis line · ${avatar.crisis_line.name} (${avatar.crisis_line.number})`
              : "Not usable yet"}
          </p>
          {state && <p className="mt-3 text-footnote text-label-secondary">{state}</p>}
        </div>

        <div className="flex shrink-0 flex-wrap gap-2">
          <Link
            href={`/upload?avatar=${avatar.id}`}
            className={controlClass({ rank: "grey", small: true })}
          >
            {avatar.photo_set_id ? "Photographs" : "Add photographs"}
          </Link>
          {avatar.callable && (
            <Link href={`/call/${avatar.id}`} className={controlClass({ rank: "filled", small: true })}>
              Call
            </Link>
          )}
        </div>
      </div>
    </li>
  );
}

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
    <Screen
      title="The people you have added."
      lede="Each one was described by you, and built from what you gave us of them."
      measure="wide"
    >
      {avatars === null && <p className="text-subhead text-label-secondary">One moment.</p>}

      {avatars?.length === 0 && (
        <Notice title="Nobody here yet.">
          You will need their name, a few words about who they were, and photographs or a short
          video of them.
        </Notice>
      )}

      {avatars && avatars.length > 0 && (
        <ul className="settle">
          {avatars.map((avatar) => (
            <Row key={avatar.id} avatar={avatar} />
          ))}
        </ul>
      )}

      <div className="mt-10">
        <Link href="/avatars/new" className={controlClass({ rank: "tinted" })}>
          Add someone
        </Link>
      </div>
    </Screen>
  );
}
