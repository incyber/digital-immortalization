"use client";

// What a visitor is told when this deployment is a shared demo.
//
// It is not decoration and it is not a badge. In demo mode there is one
// account and every visitor is signed into it, so a photograph of somebody's
// mother uploaded here is readable by the next stranger who opens the link.
// Nobody may find that out afterwards. The sentence says the consequence, in
// those words, rather than the mechanism — "shared account" is a fact about
// the server; "anyone with this link can see what you upload" is the fact
// about them.
//
// Four decisions, each because the alternative fails somebody:
//
//   It is in the root layout, so it is on every screen. A warning that appears
//   only on the entry page is a warning missed by anyone who arrives at the
//   upload screen from a link.
//
//   There is no dismiss control. It is not an interruption to be acknowledged;
//   it is a standing condition, true for as long as the page is open.
//
//   It renders nothing until the gateway has answered. Guessing would mean
//   either a warning that flashes on a private deployment or, far worse, a
//   private-looking page for the first second of a shared one.
//
//   It never renders when the flag is off, and asks once. The default
//   deployment must look exactly as it did before this file existed.

import { useSharedDemo } from "@/lib/demo";

export function DemoBanner() {
  const shared = useSharedDemo();

  if (!shared) return null;

  return (
    <aside
      role="note"
      // Sticky rather than fixed: it takes its own space at the top of the
      // document instead of sitting over the first line of whatever is there.
      className="sticky top-0 z-50 border-b border-separator-opaque bg-surface-secondary
                 px-6 py-3 text-center"
    >
      <p className="text-subhead font-semibold text-label">
        This is a shared demo. Everyone who opens this link uses the same account.
      </p>
      <p className="mt-1 text-footnote text-label-secondary">
        Anything you upload here — photographs, recordings, anything you write about
        someone — can be seen and deleted by any other visitor. Do not put a real person
        into it.
      </p>
    </aside>
  );
}
