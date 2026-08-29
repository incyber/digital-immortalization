// Progress.
//
// A hairline bar, a sentence, and a percentage only where the percentage is
// real. Long jobs get a line telling somebody they may close the page, which
// is the actual question they are sitting there asking.

export function Progress({
  value,
  label,
  detail,
}: {
  /** 0 to 1. Never let a caller pass a percentage in here by accident. */
  value: number;
  label: string;
  detail?: string;
}) {
  // A caller that has lost track of its number gets 0 rather than a bar
  // labelled NaN, which is the one thing worse than no number at all.
  const percent = Number.isFinite(value) ? Math.round(Math.min(1, Math.max(0, value)) * 100) : 0;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-subhead text-label">{label}</span>
        <span className="text-footnote tabular-nums text-label-secondary">{percent}%</span>
      </div>

      <div
        role="progressbar"
        aria-label={label}
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        className="mt-2 h-1 w-full overflow-hidden rounded-full bg-fill"
      >
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(2, percent)}%` }}
        />
      </div>

      {detail && <p className="mt-2 text-footnote text-label-secondary">{detail}</p>}
    </div>
  );
}
