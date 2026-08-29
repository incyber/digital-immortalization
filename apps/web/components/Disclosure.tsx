// Persistent synthetic-media disclosure.
//
// Rendered before the call connects and kept on screen for its whole
// duration. Not dismissible, and deliberately not styled as a cookie banner:
// the statutes that make consent necessary contemplate a viewer who knows
// throughout that they are talking to a recreation, not one who clicked past
// a notice once at signup.
//
// Two statements, both permanent. The first is that this is a recreation.
// The second, when a likeness has been built for this person, is how much of
// it was generated rather than photographed — the server's own wording, never
// rephrased here, and never dropped because a different renderer ended up
// drawing it.
//
// It is calm on purpose. A warning-coloured strip above someone's mother's
// face would be its own small violence, and a person who has been told
// gently still knows. What makes it unmissable is that it is at the top of
// the screen, in full-strength label colour, and never goes away.

export function Disclosure({ text, detail }: { text: string; detail?: string | null }) {
  return (
    <div
      role="note"
      className="flex w-full flex-col items-center gap-2 border-b border-separator
                 bg-surface-secondary px-6 py-3 text-center"
    >
      <div className="flex items-center gap-3">
        <svg
          width="15"
          height="15"
          viewBox="0 0 16 16"
          aria-hidden="true"
          fill="none"
          className="shrink-0 text-label-secondary"
        >
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.25" />
          <path d="M8 7.25v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="4.75" r="0.9" fill="currentColor" />
        </svg>
        <p className="text-footnote text-label">{text}</p>
      </div>

      {/* How much of the likeness was generated rather than photographed, in
          the sentence the build itself wrote, shown verbatim. It is the
          second line rather than the first because it is the more specific
          claim, and it is here rather than on a panel somebody opens because
          a family reads it once and then looks at the face for an hour. */}
      {detail && (
        <p className="max-w-2xl text-pretty text-caption text-label-secondary">{detail}</p>
      )}
    </div>
  );
}
