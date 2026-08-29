// A labelled field.
//
// Label above, control in the middle, one line of help beneath. Help is a
// sentence, not a warning: somebody filling this in has just lost a person and
// should never be told off by a form.

import type { ReactNode } from "react";

export function Field({
  label,
  help,
  optional = false,
  children,
}: {
  label: string;
  help?: ReactNode;
  /** Marked quietly, so that what is required is obvious without asterisks. */
  optional?: boolean;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 flex items-baseline gap-2">
        <span className="text-subhead font-medium text-label">{label}</span>
        {optional && <span className="text-footnote text-label-secondary">Optional</span>}
      </span>
      {children}
      {help && <span className="mt-2 block text-footnote text-label-secondary">{help}</span>}
    </label>
  );
}

/** The same thing where the control is not a single element a label can wrap. */
export function FieldGroup({
  label,
  help,
  optional = false,
  children,
}: {
  label: string;
  help?: ReactNode;
  optional?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="block">
      <span className="mb-2 flex items-baseline gap-2">
        <span className="text-subhead font-medium text-label">{label}</span>
        {optional && <span className="text-footnote text-label-secondary">Optional</span>}
      </span>
      {children}
      {help && <span className="mt-2 block text-footnote text-label-secondary">{help}</span>}
    </div>
  );
}
