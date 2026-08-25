"use client";

// Photo upload.
//
// Two things this page does that a plain file input would not:
//
//   It shows the shot list before anything is uploaded, fetched from the
//   gateway rather than hardcoded here, so what somebody is asked for and
//   what is enforced cannot drift apart.
//
//   It reports a per-image verdict as each upload lands. Someone photographing
//   a person who has died cannot go back and retake anything, so telling them
//   at the end which twelve pictures were unusable is far worse than telling
//   them one at a time while they still have the folder open.

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type PhotoSet, type Requirements } from "@/lib/gateway";

const REASON_TEXT: Record<string, string> = {
  "resolution below 512px on the short edge": "Too small",
  "no face detected": "No face found",
  "more than one face in frame": "More than one person",
  "too blurry or heavily upscaled": "Blurry",
  "face occupies too little of the frame": "Face too small in frame",
};

export default function Upload() {
  const router = useRouter();
  const [requirements, setRequirements] = useState<Requirements | null>(null);
  const [set, setSet] = useState<PhotoSet | null>(null);
  const [setId, setSetId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    (async () => {
      try {
        await api.me();
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) {
          router.replace("/signin");
          return;
        }
      }
      try {
        setRequirements(await api.requirements());
        const created = await api.createPhotoSet();
        setSetId(created.id);
        setSet(await api.readPhotoSet(created.id));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not start an upload");
      }
    })();
  }, [router]);

  const refresh = useCallback(async (id: string) => {
    setSet(await api.readPhotoSet(id));
  }, []);

  async function onFiles(files: FileList | null) {
    if (!files || !setId) return;
    setError(null);
    setUploading(files.length);

    // Sequential rather than parallel: each upload runs face detection on the
    // server, and twenty-five at once would queue anyway while making the
    // per-image feedback arrive in a useless order.
    for (const file of Array.from(files)) {
      try {
        await api.uploadPhoto(setId, file);
      } catch (e) {
        setError(e instanceof Error ? e.message : `Could not upload ${file.name}`);
      }
      setUploading((n) => n - 1);
      await refresh(setId);
    }
    if (input.current) input.current.value = "";
  }

  async function check() {
    if (!setId) return;
    setSet(await api.evaluate(setId));
  }

  async function startTraining() {
    if (!setId) return;
    try {
      const job = await api.train(setId);
      setJobId(job.job_id);
      setJobStatus(job.status);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start training");
    }
  }

  // Poll while a run is in flight. Training takes tens of minutes on a hosted
  // provider, so this is a status page, not a progress bar to wait in front of.
  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const job = await api.job(jobId);
        setJobStatus(job.status);
        if (job.status === "succeeded" || job.status === "failed") clearInterval(timer);
      } catch {
        // A failed poll is not a failed run; keep polling.
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [jobId]);

  const accepted = set?.photos.filter((p) => p.accepted) ?? [];
  const halfBody = accepted.filter((p) => p.half_body).length;
  const ready = set?.status === "ready";

  return (
    <main className="min-h-dvh bg-neutral-950 px-6 py-10 text-neutral-100">
      <div className="mx-auto max-w-4xl space-y-8">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight">Photographs</h1>
          {requirements && (
            <p className="mt-1.5 text-sm text-neutral-400">
              {requirements.recommended_min}–{requirements.recommended_max} photographs give
              the best likeness. At least {requirements.minimum} are needed, including{" "}
              {requirements.minimum_half_body} showing head and shoulders or more.
            </p>
          )}
        </header>

        {requirements && (
          <section className="grid gap-6 rounded-xl border border-white/10 bg-white/5 p-5 sm:grid-cols-2">
            <div>
              <h2 className="text-sm font-medium text-neutral-300">What to include</h2>
              <ul className="mt-2 space-y-1 text-sm text-neutral-400">
                {requirements.shots.map((shot) => (
                  <li key={shot.label} className="flex gap-2">
                    <span className="tabular-nums text-neutral-500">{shot.count}</span>
                    <span>{shot.label}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h2 className="text-sm font-medium text-neutral-300">Each photograph</h2>
              <ul className="mt-2 space-y-1 text-sm text-neutral-400">
                {requirements.rules.map((rule) => (
                  <li key={rule}>{rule}</li>
                ))}
              </ul>
            </div>
          </section>
        )}

        <section className="space-y-3">
          <input
            ref={input}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            onChange={(e) => onFiles(e.target.files)}
            className="block w-full text-sm text-neutral-400
                       file:mr-4 file:rounded-full file:border-0 file:bg-white
                       file:px-5 file:py-2.5 file:font-medium file:text-neutral-950
                       hover:file:bg-neutral-200"
          />
          {uploading > 0 && (
            <p className="text-sm text-neutral-400">Checking {uploading} more…</p>
          )}
          {error && (
            <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}
        </section>

        {set && set.photos.length > 0 && (
          <section className="space-y-4">
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
              <span className="text-neutral-300">
                {accepted.length} usable of {set.photos.length}
              </span>
              <span className="text-neutral-400">{halfBody} half body</span>
            </div>

            <ul className="grid gap-2 sm:grid-cols-2">
              {set.photos.map((photo) => (
                <li
                  key={photo.id}
                  className={`flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-sm ${
                    photo.accepted
                      ? "border-emerald-500/20 bg-emerald-500/5"
                      : "border-amber-500/25 bg-amber-500/5"
                  }`}
                >
                  <span className="truncate text-neutral-300">{photo.filename}</span>
                  <span className="shrink-0 text-xs text-neutral-400">
                    {photo.accepted
                      ? photo.half_body
                        ? "half body"
                        : "portrait"
                      : (photo.reasons.map((r) => REASON_TEXT[r] ?? r).join(", ") ||
                        "unusable")}
                  </span>
                </li>
              ))}
            </ul>

            {set.problems.length > 0 && (
              <ul className="space-y-1 rounded-lg border border-amber-500/25 bg-amber-500/5 px-4 py-3 text-sm text-amber-200/90">
                {set.problems.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            )}

            <div className="flex flex-wrap gap-3">
              <button
                onClick={check}
                className="rounded-full bg-white/10 px-6 py-2.5 text-sm hover:bg-white/20"
              >
                Check the set
              </button>
              <button
                onClick={startTraining}
                disabled={!ready || Boolean(jobId)}
                className="rounded-full bg-white px-6 py-2.5 text-sm font-medium text-neutral-950
                           transition hover:bg-neutral-200 disabled:opacity-40"
              >
                Build the avatar
              </button>
            </div>

            {jobStatus && (
              <p className="text-sm text-neutral-400">
                Training: <span className="text-neutral-200">{jobStatus}</span>
                {jobStatus === "running" && " — this takes a while, you can close this page."}
              </p>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
