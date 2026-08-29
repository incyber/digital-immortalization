// Persistent synthetic-media disclosure.
//
// Rendered before the call connects and kept on screen for its whole
// duration. Not dismissible, and deliberately not styled as a cookie banner:
// the statutes that make consent necessary contemplate a viewer who knows
// throughout that they are talking to a recreation, not one who clicked past
// a notice once at signup.
//
// It is calm on purpose. A warning-coloured strip above someone's mother's
// face would be its own small violence, and a person who has been told
// gently still knows. What makes it unmissable is that it is at the top of
// the screen, in full-strength label colour, and never goes away.

export function Disclosure({ text }: { text: string }) {
  return (
    <div
      role="note"
      className="flex w-full items-center justify-center gap-3 border-b border-separator
                 bg-surface-secondary px-6 py-3 text-center"
    >
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
  );
}
