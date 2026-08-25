"use client";

// Entry point. Sends people wherever they are in the flow rather than
// presenting a character: this application has none of its own.

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { api, ApiError } from "@/lib/gateway";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    (async () => {
      try {
        await api.me();
        router.replace("/avatars");
      } catch (e: unknown) {
        router.replace(e instanceof ApiError && e.status === 401 ? "/signin" : "/signin");
      }
    })();
  }, [router]);

  return (
    <main className="flex h-dvh items-center justify-center bg-neutral-950 text-neutral-500">
      <p className="text-sm">Loading…</p>
    </main>
  );
}
