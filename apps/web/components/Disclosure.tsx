// Persistent synthetic-media disclosure.
//
// Rendered before the call connects and kept on screen for its whole
// duration. Not dismissible, and deliberately not styled as a cookie banner:
// the statutes that make consent necessary contemplate a viewer who knows
// throughout that they are talking to a recreation, not one who clicked past
// a notice once at signup.

export function Disclosure({ text }: { text: string }) {
  return (
    <div
      role="note"
      className="w-full border-b border-amber-500/30 bg-amber-500/10 px-4 py-2.5
                 text-center text-sm text-amber-200/90 backdrop-blur"
    >
      {text}
    </div>
  );
}
