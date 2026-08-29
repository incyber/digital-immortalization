"use client";

// Photographs and video.
//
// Three things this page does that a plain file input would not:
//
//   It reports a per-image verdict as each upload lands. Someone photographing
//   a person who has died cannot go back and retake anything, so telling them
//   at the end which twelve pictures were unusable is far worse than telling
//   them one at a time while they still have the folder open. The ones that
//   worked are counted; only the ones that did not are listed, because those
//   are the only ones anybody can act on.
//
//   It checks the set after every upload, so nobody is asked to press a button
//   whose only job is to ask the server a question the server already knows
//   the answer to. What used to be four buttons is now one.
//
//   It states, at the moment the likeness appears, how much of it was measured
//   from what they uploaded and how much was generated. A family whose father
//   is 40% invented is told so on the screen where they first see him, not in
//   terms nobody reads.

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button, controlClass } from "@/components/ui/Button";
import { FilePicker } from "@/components/ui/FilePicker";
import { Notice } from "@/components/ui/Notice";
import { Progress } from "@/components/ui/Progress";
import { Screen } from "@/components/ui/Screen";
import {
  api,
  ApiError,
  type Avatar,
  type PhotoSet,
  type Requirements,
  type SplatJob,
  type SplatRefusal,
  type SplatRoute,
} from "@/lib/gateway";

const REASON_TEXT: Record<string, string> = {
  "resolution below 512px on the short edge": "Too small",
  "no face detected": "No face found",
  "more than one face in frame": "More than one person",
  "the face is not sharp enough": "Face not sharp",
  "face occupies too little of the frame": "Face too small in frame",
};

/** One quiet line, used for every count on this page. */
function Line({ children }: { children: React.ReactNode }) {
  return <p className="text-subhead text-label-secondary">{children}</p>;
}

export default function Upload() {
  const router = useRouter();
  const [requirements, setRequirements] = useState<Requirements | null>(null);
  const [set, setSet] = useState<PhotoSet | null>(null);
  const [setId, setSetId] = useState<string | null>(null);
  // Which person this upload is for, taken from the link that brought them
  // here. Without it there is nothing to attach a finished likeness to.
  const [avatarId, setAvatarId] = useState<string | null>(null);
  const [avatar, setAvatar] = useState<Avatar | null>(null);
  const [uploading, setUploading] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [builtAvatarId, setBuiltAvatarId] = useState<string | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [videoStatus, setVideoStatus] = useState<string | null>(null);

  // The 3D likeness. Kept apart from the training job above because it is a
  // different build with a different floor: a short clip that cannot train a
  // LoRA can still be reconstructed into a splat, so the two start and refuse
  // independently even though one button now begins both.
  const [splatJobId, setSplatJobId] = useState<string | null>(null);
  const [splatStatus, setSplatStatus] = useState<string | null>(null);
  const [splatProgress, setSplatProgress] = useState(0);
  const [splatRoute, setSplatRoute] = useState<SplatRoute | null>(null);
  const [splatReasoning, setSplatReasoning] = useState<string | null>(null);
  const [splatResult, setSplatResult] = useState<SplatJob | null>(null);
  const [splatRefusal, setSplatRefusal] = useState<SplatRefusal | null>(null);
  const [splatError, setSplatError] = useState<string | null>(null);
  const videoInput = useRef<HTMLInputElement>(null);
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

        // The avatar this upload belongs to arrives in the query string —
        // /avatars and /avatars/new both link here that way. Attaching the
        // set to it is what makes a build possible at all: both the identity
        // training and the likeness refuse a set that belongs to nobody,
        // because neither has anywhere to put what it produces.
        //
        // Read off the location rather than through useSearchParams so this
        // page needs no suspense boundary; the effect is client-only already.
        const linked = new URLSearchParams(window.location.search).get("avatar");
        if (linked) {
          setAvatarId(linked);
          await api.attachPhotoSet(linked, created.id);
          // Only so the page can say their name. A failure here is not worth
          // stopping an upload over.
          api.readAvatar(linked).then(setAvatar).catch(() => {});
        }

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

    // Evaluated here rather than behind a separate button. A set that is
    // ready should say so on its own.
    try {
      setSet(await api.evaluate(setId));
    } catch {
      // Leave the per-image verdicts standing; the explicit button remains.
    }
  }

  // A clip is usually a better source than an album: twenty seconds of someone
  // talking holds more angles and mouth positions than most families have in
  // stills. The server takes the frames, so this uploads once and waits.
  async function onVideo(files: FileList | null) {
    if (!files || !files.length || !setId) return;
    setError(null);
    setVideoStatus("Reading the clip. This takes a moment.");

    try {
      const result = await api.uploadVideo(setId, files[0]);
      setVideoStatus(
        `${result.accepted} usable of ${result.frames_examined} frames taken from the clip.`,
      );
      await refresh(setId);
      setSet(await api.evaluate(setId));
    } catch (e) {
      setVideoStatus(null);
      setError(e instanceof Error ? e.message : "Could not read that video");
    }
    if (videoInput.current) videoInput.current.value = "";
  }

  // Re-runs the current checks over what is already uploaded. Offered only
  // when something was rejected, because that is the only time it can change
  // an answer.
  async function recheck() {
    if (!setId) return;
    setError(null);
    setSet(await api.revalidate(setId));
  }

  async function startTraining() {
    if (!setId) return;
    try {
      const job = await api.train(setId);
      setJobId(job.job_id);
      setJobStatus(job.status);
      setProgress(0.02); // visible immediately, so the click clearly landed
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start training");
    }
  }

  async function startLikeness() {
    if (!setId) return;
    setSplatError(null);
    setSplatRefusal(null);
    try {
      const started = await api.buildSplat(setId);
      if (!started.buildable) {
        // Not an error, and deliberately not thrown. The customer is being
        // told what else is needed, in counts they can check against their
        // own folder, which is something they can act on.
        setSplatRefusal(started);
        return;
      }
      setSplatJobId(started.job_id);
      setSplatStatus("running");
      setSplatRoute(started.route);
      setSplatReasoning(started.reasoning);
      setSplatProgress(0.02); // visible immediately, so the click clearly landed
    } catch (e) {
      setSplatError(e instanceof Error ? e.message : "Could not start the likeness");
    }
  }

  // Poll the likeness build. The bar is an estimate from elapsed time — no
  // splat optimiser reports its own progress — and it never reaches the end
  // until the artefact exists.
  useEffect(() => {
    if (!splatJobId) return;
    const timer = setInterval(async () => {
      try {
        const job = await api.splatJob(splatJobId);
        setSplatStatus(job.status);
        // Never let the bar go backwards on an out-of-order poll, and never
        // let a response without a number in it turn the bar into NaN.
        setSplatProgress((p) => (Number.isFinite(job.progress) ? Math.max(p, job.progress) : p));
        if (job.error) setSplatError(job.error);
        if (job.route) setSplatRoute(job.route);
        if (job.reasoning) setSplatReasoning(job.reasoning);
        if (job.status === "succeeded") setSplatResult(job);
        if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
          clearInterval(timer);
        }
      } catch {
        // A failed poll is not a failed build; keep polling.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [splatJobId]);

  // Poll while a run is in flight. Training takes tens of minutes on a hosted
  // provider, so this is a status page, not a progress bar to wait in front of.
  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const job = await api.job(jobId);
        setJobStatus(job.status);
        // Never let the bar go backwards: a poll that arrives out of order
        // otherwise makes it jump about. A response with no progress in it
        // leaves the bar where it was rather than blanking it.
        setProgress((p) => (Number.isFinite(job.progress) ? Math.max(p, job.progress) : p));
        if (job.avatar_id) setBuiltAvatarId(job.avatar_id);
        if (job.error) setJobError(job.error);
        if (job.status === "succeeded" || job.status === "failed") {
          clearInterval(timer);
        }
      } catch {
        // A failed poll is not a failed run; keep polling.
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);

  const photos = set?.photos ?? [];
  const accepted = photos.filter((p) => p.accepted);
  const rejected = photos.filter((p) => !p.accepted);
  const splatRunning = splatStatus === "running" || splatStatus === "queued";
  const measured =
    splatResult?.measured_fraction === null || splatResult?.measured_fraction === undefined
      ? null
      : Math.round(splatResult.measured_fraction * 100);
  const generated = measured === null ? null : 100 - measured;
  const halfBody = accepted.filter((p) => p.half_body).length;
  const ready = set?.status === "ready";
  // What is still short, in counts, and what the server itself says is
  // stopping the build.
  //
  // The count rows come from the requirement list; the sentence comes from
  // `problems`, which the gateway writes only for the failures that actually
  // hold a set back. The requirement rows carry no usable flag for that —
  // the gateway does not serialise one — so nothing here guesses which of
  // them is decisive. Torso coverage, for one, changes what the avatar looks
  // like rather than whether it can be made.
  const stillShort = (set?.requirements ?? []).filter((r) => !r.met);
  const problems = set && !ready ? (set.problems ?? []) : [];

  // One action, two builds. Training needs a set the server calls ready; the
  // likeness has its own, lower floor, so a short clip can still produce
  // something to look at while more photographs are being found.
  const canTrain = ready && !jobId;
  const canLikeness = Boolean(avatarId) && accepted.length > 0 && !splatRunning && !splatJobId;
  const canBuild = canTrain || canLikeness;

  async function build() {
    setError(null);
    if (canTrain) await startTraining();
    if (canLikeness) await startLikeness();
  }

  return (
    <Screen
      back={{ href: "/avatars", label: "Back" }}
      title={avatar ? `Photographs of ${avatar.display_name}.` : "Their photographs."}
      lede={
        requirements
          ? `A short video works best. Failing that, ${requirements.recommended_min} to ${requirements.recommended_max} photographs — ${requirements.minimum} is the fewest that can be used.`
          : undefined
      }
      measure="wide"
    >
      <div className="space-y-12">
        {/* A clip, first, because it is the better source. The renderer
            animates a mouth onto this footage, so the head in the result
            moves because the head in the recording moved. Photographs can
            only be given synthesised motion, which is a harder problem with a
            visibly worse answer. */}
        <section>
          <h2 className="text-title-3 text-label">A short video</h2>
          <p className="mt-2 max-w-prose text-subhead text-label-secondary">
            Twenty to sixty seconds of them talking to camera, head and shoulders in frame. Their
            real movement, blinks and posture are kept; only the mouth changes as they speak.
          </p>
          <div className="mt-6">
            <FilePicker
              rank="tinted"
              label="Choose a video"
              hint="MP4, MOV, WebM or MKV."
              dropHint="You can drop one here instead."
              accept="video/mp4,video/quicktime,video/webm,video/x-matroska"
              inputRef={videoInput}
              onFiles={onVideo}
            />
          </div>
          {videoStatus && <p className="mt-4 text-subhead text-label">{videoStatus}</p>}
        </section>

        <section className="border-t border-separator pt-12">
          <h2 className="text-title-3 text-label">Photographs</h2>
          <p className="mt-2 max-w-prose text-subhead text-label-secondary">
            Use whatever exists. Close portraits are fine, and different days, outfits and
            lighting help.
          </p>
          <div className="mt-6">
            <FilePicker
              label="Choose photographs"
              hint="JPEG, PNG or WebP."
              dropHint="A whole folder can be dropped here at once."
              accept="image/jpeg,image/png,image/webp"
              multiple
              inputRef={input}
              onFiles={onFiles}
            />
          </div>
          {uploading > 0 && (
            <p className="mt-4 text-subhead text-label-secondary" aria-live="polite">
              Checking {uploading} more…
            </p>
          )}

          {requirements && (
            <details className="group mt-6">
              <summary
                className="flex cursor-pointer list-none items-center gap-2 text-subhead
                           text-accent transition-opacity duration-200 hover:opacity-70
                           [&::-webkit-details-marker]:hidden"
              >
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 12 12"
                  aria-hidden="true"
                  fill="none"
                  className="transition-transform duration-200 group-open:rotate-90"
                >
                  <path
                    d="M4 2l4 4-4 4"
                    stroke="currentColor"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                What works best
              </summary>
              <div className="mt-6 grid gap-8 sm:grid-cols-2">
                <div>
                  <h3 className="text-footnote font-semibold text-label">What to include</h3>
                  <ul className="mt-3 space-y-2">
                    {requirements.shots.map((shot) => (
                      <li key={shot.label} className="flex gap-3 text-subhead text-label-secondary">
                        <span className="tabular-nums text-label-tertiary">{shot.count}</span>
                        <span>{shot.label}</span>
                      </li>
                    ))}
                  </ul>
                  {requirements.note && (
                    <p className="mt-4 text-footnote text-label-secondary">{requirements.note}</p>
                  )}
                </div>
                <div>
                  <h3 className="text-footnote font-semibold text-label">Each photograph</h3>
                  <ul className="mt-3 space-y-2">
                    {requirements.rules.map((rule) => (
                      <li key={rule} className="text-subhead text-label-secondary">
                        {rule}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </details>
          )}

          {error && (
            <Notice tone="problem" role="alert" className="mt-6">
              {error}
            </Notice>
          )}
        </section>

        {photos.length > 0 && (
          <section className="settle border-t border-separator pt-12">
            <h2 className="text-title-3 text-label">
              {accepted.length} usable of {photos.length}
            </h2>
            <Line>
              {halfBody} with the chest in frame
              {set?.framing_label ? ` · framed at ${set.framing_label}` : ""}
            </Line>

            {problems.length > 0 && (
              <Notice tone="attention" className="mt-6" title="Not quite enough yet.">
                {problems.map((problem) => (
                  <p key={problem} className="first-letter:uppercase">
                    {problem}.
                  </p>
                ))}
              </Notice>
            )}

            {stillShort.length > 0 && (
              <ul className="mt-6">
                {stillShort.map((r) => (
                  <li key={r.key} className="border-t border-separator py-3 first:border-t-0 first:pt-0">
                    <div className="flex items-baseline justify-between gap-6">
                      <span className="text-subhead text-label">{r.label}</span>
                      <span className="shrink-0 text-subhead tabular-nums text-label-secondary">
                        {r.current} of {r.target}
                      </span>
                    </div>
                    {r.hint && <p className="mt-1 text-footnote text-label-secondary">{r.hint}</p>}
                  </li>
                ))}
              </ul>
            )}

            {/* Only the ones that cannot be used are listed. A family does not
                need thirty rows confirming that a photograph worked. */}
            {rejected.length > 0 && (
              <div className="mt-8">
                <h3 className="text-headline text-label">
                  {rejected.length === 1
                    ? "One photograph cannot be used"
                    : `${rejected.length} photographs cannot be used`}
                </h3>
                <ul className="mt-4">
                  {rejected.map((photo) => (
                    <li
                      key={photo.id}
                      className="flex items-baseline justify-between gap-6 border-t
                                 border-separator py-3 first:border-t-0 first:pt-0"
                    >
                      <span className="truncate text-subhead text-label">{photo.filename}</span>
                      <span className="shrink-0 text-footnote text-label-secondary">
                        {photo.reasons.map((r) => REASON_TEXT[r] ?? r).join(", ") || "unusable"}
                      </span>
                    </li>
                  ))}
                </ul>
                <div className="mt-4">
                  <Button rank="plain" small className="-ml-2" onClick={recheck}>
                    Check these again
                  </Button>
                </div>
              </div>
            )}

            {accepted.length > 0 && (
              <details className="group mt-8">
                <summary
                  className="flex cursor-pointer list-none items-center gap-2 text-subhead
                             text-accent transition-opacity duration-200 hover:opacity-70
                             [&::-webkit-details-marker]:hidden"
                >
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 12 12"
                    aria-hidden="true"
                    fill="none"
                    className="transition-transform duration-200 group-open:rotate-90"
                  >
                    <path
                      d="M4 2l4 4-4 4"
                      stroke="currentColor"
                      strokeWidth="1.75"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Every photograph
                </summary>
                <ul className="mt-4">
                  {accepted.map((photo) => (
                    <li
                      key={photo.id}
                      className="flex items-baseline justify-between gap-6 border-t
                                 border-separator py-3 first:border-t-0 first:pt-0"
                    >
                      <span className="truncate text-subhead text-label-secondary">
                        {photo.filename}
                      </span>
                      <span className="shrink-0 text-footnote text-label-secondary">
                        {photo.half_body ? "chest in frame" : "portrait"}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        )}

        {!avatarId && photos.length > 0 && (
          /* Reached without a person to build. Said plainly rather than left
             as a disabled button with no explanation. */
          <Notice tone="attention" title="This upload has nowhere to go yet.">
            Start from their card so the likeness can be attached to them. Open{" "}
            <Link href="/avatars" className="text-accent underline-offset-4 hover:underline">
              the people you have added
            </Link>{" "}
            and choose Add photographs.
          </Notice>
        )}

        {/* One action. It starts whichever of the two builds it can: the
            likeness has a lower floor than the training, so a short clip can
            produce something to look at while more photographs are found. */}
        {photos.length > 0 && (canBuild || (!jobId && !splatJobId)) && (
          <section className="border-t border-separator pt-12">
            <Button rank="filled" onClick={build} disabled={!canBuild}>
              {canTrain ? "Build them" : "Build the likeness"}
            </Button>
            <p className="mt-4 text-footnote text-label-secondary">
              {canTrain
                ? "Takes a while. You can close this page — it keeps building."
                : canLikeness
                  ? "The likeness can be built from what is here. The rest of them needs more photographs, and can be built once you have added them."
                  : avatarId
                    ? "Add photographs or a video, and they can be built."
                    : "Nothing can be built until this upload belongs to somebody."}
            </p>
          </section>
        )}

        {jobStatus && (
          <section className="settle border-t border-separator pt-12">
            {/* A finished job is a sentence, not a bar sitting at the end of
                its track. The bar is only there while there is something
                left to wait for. */}
            {jobStatus === "succeeded" || jobStatus === "failed" ? (
              <p className="text-headline text-label">
                {jobStatus === "succeeded" ? "They are ready." : "Building stopped."}
              </p>
            ) : (
              <Progress
                value={progress}
                label="Building them"
                detail="You can close this page — it keeps building."
              />
            )}

            {jobError && (
              <Notice tone="problem" role="alert" className="mt-6">
                {jobError}
              </Notice>
            )}

            {jobStatus === "succeeded" && builtAvatarId && (
              <div className="mt-6">
                <Link href={`/call/${builtAvatarId}`} className={controlClass({ rank: "filled" })}>
                  Talk to them
                </Link>
              </div>
            )}
          </section>
        )}

        {/* Guidance, never an error style. Somebody who could only find two
            photographs of their father has not done anything wrong, and the
            counts are theirs so they can check them against the folder still
            open beside this page. */}
        {splatRefusal && (
          <Notice tone="attention" title="A little more is needed for the likeness.">
            <p>{splatRefusal.guidance}</p>
            <ul className="mt-3 space-y-1">
              {splatRefusal.missing.map((item) => (
                <li key={item} className="flex gap-3">
                  <span className="text-label-tertiary">·</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
            <p className="mt-3 text-footnote text-label-secondary">{splatRefusal.reasoning}</p>
          </Notice>
        )}

        {splatStatus && (
          <section className="settle border-t border-separator pt-12">
            {splatRunning ? (
              <Progress
                value={splatProgress}
                label="Building the likeness"
                detail="You can close this page — it keeps building."
              />
            ) : (
              <p className="text-headline text-label">
                {splatStatus === "succeeded"
                  ? splatRoute === "reconstruct"
                    ? "The likeness is built, from your video."
                    : "The likeness is built, from your photographs."
                  : splatStatus === "cancelled"
                    ? "Building the likeness was stopped."
                    : "Building the likeness stopped."}
              </p>
            )}

            {/* Which route built it, and why. The customer chose neither; what
                they were able to upload decided it, and saying so is what
                stops "why does this one look different" being a mystery six
                months later. */}
            {splatReasoning && (
              <p className="mt-4 text-subhead text-label-secondary">{splatReasoning}</p>
            )}

            {splatError && (
              <Notice tone="problem" role="alert" className="mt-6">
                {splatError}
              </Notice>
            )}

            {/* The disclosure, at the moment the likeness appears. Not a
                footnote and not in terms: a photographs-only build invents the
                angles no camera covered, and the family reading this screen is
                the one entitled to know that first. */}
            {splatResult?.disclosure && (
              <div className="group-surface mt-8 p-6">
                <p className="text-body text-label">{splatResult.disclosure}</p>

                {measured !== null && generated !== null && (
                  <div className="mt-6">
                    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-fill">
                      <div className="h-full bg-accent" style={{ width: `${measured}%` }} />
                      <div className="h-full bg-orange" style={{ width: `${generated}%` }} />
                    </div>
                    <p className="mt-3 text-footnote tabular-nums text-label-secondary">
                      <span className="text-label">{measured}% measured</span> from what you
                      uploaded · <span className="text-label">{generated}% generated</span>
                    </p>
                  </div>
                )}

                {splatResult.concerns.length > 0 && (
                  <ul className="mt-4 space-y-1">
                    {splatResult.concerns.map((concern) => (
                      <li key={concern} className="flex gap-3 text-footnote text-label-secondary">
                        <span className="text-label-tertiary">·</span>
                        <span>{concern}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {splatResult.gaussians > 0 && (
                  <p className="mt-4 text-footnote tabular-nums text-label-secondary">
                    {splatResult.gaussians.toLocaleString()} points ·{" "}
                    {Math.round(splatResult.size_bytes / (1024 * 1024))}MB to download
                  </p>
                )}
              </div>
            )}
          </section>
        )}
      </div>
    </Screen>
  );
}
