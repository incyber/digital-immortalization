"use client";

// Entry point. Sends people wherever they are in the flow rather than
// presenting a character: this application has none of its own.
//
// Somebody signed in with nobody recreated yet goes straight to the one screen
// that asks about a person. A list with nothing in it is a step that exists
// only to be clicked past.

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { api } from "@/lib/gateway";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    (async () => {
      try {
        await api.me();
      } catch {
        router.replace("/signin");
        return;
      }
      try {
        const { avatars } = await api.listAvatars();
        router.replace(avatars.length === 0 ? "/avatars/new" : "/avatars");
      } catch {
        router.replace("/avatars");
      }
    })();
  }, [router]);

  return (
    <main className="flex h-dvh items-center justify-center bg-surface">
      <p className="text-subhead text-label-secondary">One moment.</p>
    </main>
  );
}
