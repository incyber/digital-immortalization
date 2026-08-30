"use client";

// Whether this deployment is a shared demo.
//
// Asked once per page load and shared by every caller. Two components need the
// answer — the banner across the top and the line in the footer that would
// otherwise promise a privacy this mode does not have — and they must never
// disagree with each other or ask twice.
//
// The default is false in every uncertain case: before the gateway has
// answered, and if it never does. Getting it wrong in that direction shows a
// private deployment as private, which it is. Getting it wrong the other way
// would put a "your uploads are shared" warning on a deployment where they are
// not, and teach people to ignore it.

import { useEffect, useState } from "react";
import { api } from "@/lib/gateway";

// Module scope, so a navigation between pages reuses the answer rather than
// asking the gateway again. Cleared by a full page load, which is the only
// point at which the deployment could have changed under us.
let asked: Promise<boolean> | null = null;

function isSharedDemo(): Promise<boolean> {
  asked ??= api
    .config()
    .then((config) => config.demo_mode)
    .catch(() => false);
  return asked;
}

export function useSharedDemo(): boolean {
  const [shared, setShared] = useState(false);

  useEffect(() => {
    let live = true;
    isSharedDemo().then((value) => {
      if (live) setShared(value);
    });
    return () => {
      live = false;
    };
  }, []);

  return shared;
}
