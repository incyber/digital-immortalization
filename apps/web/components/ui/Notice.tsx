// A short message that is not the content of the page.
//
// Three tones, and none of them shout. A grieving person reading that eleven
// of their photographs were unusable does not need a red box to understand
// it; they need a plain sentence and somewhere to go next. Meaning is carried
// by one line of colour and by where the message sits, never by a wash of
// colour behind the words.

import type { ReactNode } from "react";

export type NoticeTone = "quiet" | "attention" | "problem";

const RULE: Record<NoticeTone, string> = {
  quiet: "before:bg-separator-opaque",
  attention: "before:bg-orange-soft",
  problem: "before:bg-red",
};

export function Notice({
  tone = "quiet",
  title,
  children,
  className = "",
  role,
}: {
  tone?: NoticeTone;
  title?: string;
  children?: ReactNode;
  className?: string;
  role?: "status" | "alert" | "note";
}) {
  return (
    <div
      role={role}
      className={`relative rounded-xl bg-surface-secondary px-6 py-4 pl-8
                  before:absolute before:top-4 before:bottom-4 before:left-4 before:w-[3px]
                  before:rounded-full ${RULE[tone]} ${className}`}
    >
      {title && <p className="text-subhead font-semibold text-label">{title}</p>}
      {children && (
        <div className={`text-subhead text-label-secondary ${title ? "mt-2" : ""}`}>{children}</div>
      )}
    </div>
  );
}
