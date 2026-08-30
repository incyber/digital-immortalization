"use client";

// The one line in the footer that says who can see what was uploaded.
//
// It is a client component for a single reason: the sentence is false in demo
// mode. "Everything you upload stays in your account" is a promise of privacy,
// and in a shared demo there is one account and everyone is in it. Leaving the
// original line there under a banner that says the opposite would be worse
// than saying nothing — the reassuring sentence is the one people believe.

import { useSharedDemo } from "@/lib/demo";

export function AccountScope() {
  const shared = useSharedDemo();

  return (
    <p className="text-footnote text-label-secondary">
      {shared
        ? "This is a shared demo account. Anyone with the link can see what you upload."
        : "Everything you upload stays in your account."}
    </p>
  );
}
