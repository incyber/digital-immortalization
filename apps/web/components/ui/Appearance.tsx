"use client";

// The appearance choice.
//
// Light is the default because that is what the product is designed in. Dark
// is offered because people sit with this late at night, and a white screen at
// two in the morning is its own small cruelty.
//
// Rendered in a footer rather than a header: it is a preference, not a task,
// and nothing about it should compete with the page.

import { useLayoutEffect, useSyncExternalStore } from "react";
import { Segmented } from "@/components/ui/Segmented";
import {
  applyAppearance,
  readAppearance,
  restoreAppearance,
  serverAppearance,
  subscribeAppearance,
  type Appearance as Choice,
} from "@/lib/appearance";

const OPTIONS: { value: Choice; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

export function Appearance() {
  // Read from storage rather than held in state. The document already has the
  // right appearance by this point — the script in the head set it before the
  // first paint — so this control only has to agree with what is on screen.
  const choice = useSyncExternalStore(subscribeAppearance, readAppearance, serverAppearance);

  // Writes the attribute back onto the document. Only ever does anything in
  // development, where a Strict Mode remount clears what the head script set.
  useLayoutEffect(restoreAppearance, []);

  return (
    <div className="w-full max-w-[15rem]">
      <Segmented label="Appearance" value={choice} options={OPTIONS} onChange={applyAppearance} />
    </div>
  );
}
