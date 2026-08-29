"use client";

// A segmented control.
//
// Used where the options are few, short, and worth seeing at once — the units
// a height is typed in, who is authorising a recreation, which appearance the
// interface should take. A dropdown hides the answer until it is opened, and
// on a question like "who are you to this person" that matters.

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  className = "",
}: {
  value: T | null;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
  className?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={`inline-flex w-full rounded-[10px] bg-fill p-[2px] ${className}`}
    >
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onChange(option.value)}
            className={`flex-1 rounded-lg px-4 py-2 text-subhead transition-colors duration-200
                        ${
                          selected
                            ? "bg-selected font-medium text-label shadow-[0_1px_2px_rgba(0,0,0,0.08)]"
                            : "font-normal text-label-secondary hover:text-label"
                        }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
