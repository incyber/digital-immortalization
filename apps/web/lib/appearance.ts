// Light, dark, or whatever the machine is set to.
//
// The stored value is the person's choice and nothing else. "auto" is stored
// as the absence of a choice, so a machine that later changes its own setting
// is followed rather than overridden by something chosen months ago.
//
// Storage is an external system, so it is read through a subscription rather
// than copied into React state. That also means a change made in one tab
// reaches the others.

export type Appearance = "auto" | "light" | "dark";

const KEY = "appearance";

const listeners = new Set<() => void>();

/** The stored choice. A primitive, so it is safe as a store snapshot. */
export function readAppearance(): Appearance {
  try {
    const stored = localStorage.getItem(KEY);
    return stored === "light" || stored === "dark" ? stored : "auto";
  } catch {
    // Private browsing, or storage turned off. The system preference stands.
    return "auto";
  }
}

/** What the server renders, before any storage exists to read. */
export function serverAppearance(): Appearance {
  return "auto";
}

export function subscribeAppearance(onChange: () => void): () => void {
  listeners.add(onChange);
  window.addEventListener("storage", onChange);
  return () => {
    listeners.delete(onChange);
    window.removeEventListener("storage", onChange);
  };
}

/**
 * Put the stored choice back on the document.
 *
 * The script in the head does this before the first paint. React's Strict
 * Mode remounts once in development and resets the attributes it manages on
 * <html>, which clears that one, so the choice is written again after mount.
 * In production this is a no-op that agrees with what is already there.
 */
export function restoreAppearance(): void {
  const choice = readAppearance();
  if (choice === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = choice;
}

export function applyAppearance(next: Appearance): void {
  const root = document.documentElement;
  if (next === "auto") {
    delete root.dataset.theme;
  } else {
    root.dataset.theme = next;
  }
  try {
    if (next === "auto") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, next);
  } catch {
    // The choice still applies to this page; it simply will not be remembered.
  }
  // A write from this tab raises no storage event here, so tell the readers.
  listeners.forEach((notify) => notify());
}
