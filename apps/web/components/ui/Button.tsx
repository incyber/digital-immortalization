// Buttons.
//
// Three ranks and no more: filled for the one thing a screen is for, tinted
// or grey for a real alternative, plain for everything that is not an action
// so much as a way out. Destructive is its own rank because ending a call
// should never look like the same weight of decision as starting one.
//
// The class list lives in globals.css under .control so that a Link can wear
// the same button without a second definition drifting away from this one.

import type { ButtonHTMLAttributes } from "react";

export type ControlRank = "filled" | "tinted" | "grey" | "plain" | "destructive";

export function controlClass({
  rank = "filled",
  small = false,
  wide = false,
  className = "",
}: {
  rank?: ControlRank;
  small?: boolean;
  wide?: boolean;
  className?: string;
} = {}) {
  return [
    "control",
    `control-${rank}`,
    small ? "control-small" : "",
    wide ? "control-wide" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
}

export function Button({
  rank = "filled",
  small = false,
  wide = false,
  className = "",
  type = "button",
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  rank?: ControlRank;
  small?: boolean;
  wide?: boolean;
}) {
  return <button type={type} className={controlClass({ rank, small, wide, className })} {...rest} />;
}
