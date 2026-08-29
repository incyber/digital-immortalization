// The body questions, in the words a family will read.
//
// The wording lives here rather than inside either page so the two cannot
// drift into describing the same person differently. The values are the
// gateway's own vocabulary: it refuses anything outside them rather than
// reshaping it, which is why every one of these is chosen from a list and
// none of them is typed.

import type { Build, Posture, ShoulderWidth, StatedBody } from "@/lib/gateway";

export const BUILDS: { value: Build; label: string }[] = [
  { value: "slight", label: "Slight — light, small-framed" },
  { value: "average", label: "Average" },
  { value: "solid", label: "Solid — sturdy, well built" },
  { value: "heavy", label: "Heavy — a larger, fuller frame" },
];

export const SHOULDERS: { value: ShoulderWidth; label: string }[] = [
  { value: "narrow", label: "Narrow" },
  { value: "average", label: "Average" },
  { value: "broad", label: "Broad" },
];

export const POSTURES: { value: Posture; label: string }[] = [
  { value: "upright", label: "Upright — stood straight" },
  { value: "relaxed", label: "Relaxed — easy, natural" },
  { value: "stooped", label: "Stooped — leaned forward" },
];

// Matches what the gateway will accept. Wide on purpose: the point is to catch
// a slipped finger, not to argue with somebody about a height they remember.
export const MIN_HEIGHT_CM = 50;
export const MAX_HEIGHT_CM = 250;

export function toFeetInches(cm: number): { feet: number; inches: number } {
  const total = Math.round(cm / 2.54);
  return { feet: Math.floor(total / 12), inches: total % 12 };
}

export function fromFeetInches(feet: number, inches: number): number {
  return Math.round((feet * 12 + inches) * 2.54);
}

// Both units, always. Half the people using this think in centimetres and the
// other half in feet, and neither should have to do the arithmetic.
export function bothUnits(cm: number): string {
  const { feet, inches } = toFeetInches(cm);
  return `${cm} cm · ${feet} ft ${inches} in`;
}

function labelFor<T extends string>(
  options: { value: T; label: string }[],
  value: T | null,
): string | null {
  const found = options.find((o) => o.value === value);
  // Only the short word, never the explanation that helps somebody choose it.
  return found ? found.label.split(" — ")[0].toLowerCase() : null;
}

// A single quiet line for a list, or null when nothing was said. Nothing was
// said is a normal state and reads as one.
export function summarise(stated: StatedBody): string | null {
  const parts = [
    stated.height_cm === null ? null : bothUnits(stated.height_cm),
    labelFor(BUILDS, stated.build) && `${labelFor(BUILDS, stated.build)} build`,
    labelFor(SHOULDERS, stated.shoulders) && `${labelFor(SHOULDERS, stated.shoulders)} shoulders`,
    labelFor(POSTURES, stated.posture),
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}
