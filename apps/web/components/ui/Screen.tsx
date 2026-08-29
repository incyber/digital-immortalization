// The page shell.
//
// A narrow measure, a lot of space around it, and a title that is a sentence
// rather than a section heading. Everything sits in one column: there is no
// navigation to speak of, because at any moment there is only one thing this
// product is asking somebody to do.

import Link from "next/link";
import type { ReactNode } from "react";
import { Appearance } from "@/components/ui/Appearance";

/** How wide the column runs. Reading stays narrow; grids may go wide. */
const MEASURE = {
  tight: "max-w-[26rem]",
  narrow: "max-w-xl",
  wide: "max-w-3xl",
} as const;

export function Screen({
  title,
  lede,
  back,
  children,
  measure = "narrow",
  center = false,
}: {
  title?: string;
  lede?: ReactNode;
  back?: { href: string; label: string };
  children: ReactNode;
  measure?: keyof typeof MEASURE;
  /** Short screens sit in the middle of the window rather than at the top. */
  center?: boolean;
}) {
  const column = `mx-auto w-full ${MEASURE[measure]}`;

  return (
    <div className="flex min-h-dvh flex-col bg-surface">
      {back && (
        <header className="px-6 pt-6">
          <div className={column}>
            <Link
              href={back.href}
              className="-ml-2 inline-flex min-h-[44px] items-center gap-1 px-2 text-body
                         text-accent transition-opacity duration-200 hover:opacity-70"
            >
              <svg width="10" height="16" viewBox="0 0 10 16" aria-hidden="true" fill="none">
                <path
                  d="M8 1L2 8l6 7"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {back.label}
            </Link>
          </div>
        </header>
      )}

      <main
        className={`flex-1 px-6 ${back ? "pt-6" : "pt-16"} pb-16 ${
          center ? "flex flex-col justify-center" : ""
        }`}
      >
        <div className={column}>
          {(title || lede) && (
            <div className="mb-10">
              {title && <h1 className="text-large-title text-balance text-label">{title}</h1>}
              {lede && (
                <p className="mt-4 max-w-prose text-body text-pretty text-label-secondary">
                  {lede}
                </p>
              )}
            </div>
          )}
          {children}
        </div>
      </main>

      <footer className="px-6 pb-10">
        <div
          className={`${column} flex flex-wrap items-center justify-between gap-6
                      border-t border-separator pt-8`}
        >
          <p className="text-footnote text-label-secondary">
            Everything you upload stays in your account.
          </p>
          <Appearance />
        </div>
      </footer>
    </div>
  );
}
