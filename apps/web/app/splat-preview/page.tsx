"use client";

// Internal instrument — not a customer surface.
//
// This page exists to answer whether a visitor's own device can render the
// likeness, because rendering on the viewer's hardware rather than on a rented
// GPU is the difference between a call costing pennies and costing dollars.
// It is therefore a measuring instrument first: every number it shows is one
// that decision turns on, and none of them are smoothed, rounded up, or
// estimated when they are not actually known.
//
// The mount/unmount control is not a convenience. Loading, discarding and
// reloading is how a leaked WebGL context or an accumulating texture pool
// shows itself, and a browser silently kills the oldest context once a page
// holds around sixteen of them.

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { SplatStage } from "@/components/SplatStage";
import {
  DEFAULT_FRAMING,
  DEFAULT_POSE,
  formatBytes,
  idleStatus,
  probeDevice,
  type CameraFraming,
  type DeviceProfile,
  type FrameStats,
  type HeadPose,
  type SplatStatus,
} from "@/lib/splat";

// Public samples published by the renderer's authors, served with permissive
// CORS. They are all far smaller than a real capture: useful for proving the
// pipeline and for reading frame rate against splat count, useless for judging
// download or decode time at production sizes. The page says so.
const SAMPLES = [
  { label: "Robot head", url: "https://sparkjs.dev/assets/splats/robot-head.spz", size: 1153152 },
  { label: "Penguin", url: "https://sparkjs.dev/assets/splats/penguin.spz", size: 2520338 },
  { label: "Butterfly", url: "https://sparkjs.dev/assets/splats/butterfly.spz", size: 4025604 },
  { label: "Forge (scene)", url: "https://sparkjs.dev/assets/splats/forge.spz", size: 5966265 },
  { label: "Valley (scene)", url: "https://sparkjs.dev/assets/splats/valley.spz", size: 6752303 },
  { label: "Snow street (scene)", url: "https://sparkjs.dev/assets/splats/snow-street.spz", size: 9936337 },
];

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div className="tabular-nums text-sm text-neutral-100">{value}</div>
      {hint && <div className="text-[11px] text-neutral-500">{hint}</div>}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between text-xs text-neutral-400">
        <span>{label}</span>
        <span className="tabular-nums text-neutral-300">
          {value}
          {unit}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-emerald-400"
      />
    </label>
  );
}

export default function SplatPreview() {
  const [inputUrl, setInputUrl] = useState(SAMPLES[0].url);
  const [activeUrl, setActiveUrl] = useState<string | null>(null);
  // Bumped on every load so remounting is a real teardown and rebuild rather
  // than React reusing the existing viewer.
  const [cycle, setCycle] = useState(0);

  const [pose, setPose] = useState<HeadPose>(DEFAULT_POSE);
  const [framing, setFraming] = useState<CameraFraming>(DEFAULT_FRAMING);

  const [status, setStatus] = useState<SplatStatus>(idleStatus);
  const [frameStats, setFrameStats] = useState<FrameStats | null>(null);
  const [device, setDevice] = useState<DeviceProfile | null>(null);

  // Worst frame rate seen since this asset was loaded. A mean hides the stalls
  // that make a call feel broken, and the stalls are what a mid-range phone
  // will produce first.
  const worstFpsRef = useRef<number | null>(null);
  const [worstFps, setWorstFps] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    probeDevice().then((profile) => {
      if (!cancelled) setDevice(profile);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  function load(url: string) {
    worstFpsRef.current = null;
    setWorstFps(null);
    setFrameStats(null);
    setStatus(idleStatus());
    setActiveUrl(url);
    setCycle((n) => n + 1);
  }

  function unload() {
    setActiveUrl(null);
    setFrameStats(null);
    setStatus(idleStatus());
    // Cleared with the rest: a worst-second measured against the asset just
    // discarded, sitting beside a row of blanks, is the one tile that would
    // still be asserting something after there is nothing left to measure.
    worstFpsRef.current = null;
    setWorstFps(null);
  }

  function handleFrameStats(next: FrameStats) {
    setFrameStats(next);
    // Only counted once the asset is on screen: the frames rendered while an
    // empty stage waits for a download would flatter the number.
    if (status.phase !== "ready") return;
    if (worstFpsRef.current === null || next.fps < worstFpsRef.current) {
      worstFpsRef.current = next.fps;
      setWorstFps(next.fps);
    }
  }

  const sampleNote = useMemo(() => {
    const largest = Math.max(...SAMPLES.map((s) => s.size));
    // Two separate caveats, and the second is the one that gets forgotten.
    // Size limits what the timings mean; subject limits what the *picture*
    // means. Every reachable public sample is an object, an animal or a room —
    // there is no face among them — and skin, eyes and hair are the hard case
    // for this technique. A robot head rendering beautifully says nothing
    // about whether somebody's father will be recognisable.
    return `Largest reachable public sample is ${formatBytes(largest)}. A production capture is tens to hundreds of megabytes, so download and unpack timings here are not representative — frame rate against splat count is. None of these samples is a person: they answer whether this device can render a splat, not whether a face holds up.`;
  }, []);

  const throughput =
    status.downloadMs && status.downloadMs > 0 && status.bytesLoaded > 0
      ? `${formatBytes((status.bytesLoaded / status.downloadMs) * 1000)}/s`
      : null;

  return (
    <main className="min-h-dvh bg-neutral-950 px-6 py-8 text-neutral-100">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-amber-300">
              Internal — not a customer page
            </span>
            <Link href="/" className="text-xs text-neutral-500 underline-offset-4 hover:underline">
              Back
            </Link>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">Device-side splat rendering</h1>
          <p className="max-w-3xl text-sm text-neutral-400">
            Measures whether a likeness can be rendered on the visitor&rsquo;s own hardware. Load an
            asset, read the frame rate, and move the head to confirm the pose plumbing reaches the
            renderer. Numbers are reported as measured; anything genuinely unknown is left blank
            rather than estimated.
          </p>
        </header>

        {device && !device.supported && (
          <p className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {device.reason}
          </p>
        )}

        <section className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <input
              type="url"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              placeholder="https://…/likeness.spz"
              aria-label="Splat asset URL"
              spellCheck={false}
              className="min-w-0 flex-1 rounded-full border border-white/10 bg-white/5 px-4 py-2.5
                         text-sm text-neutral-100 placeholder:text-neutral-600
                         focus:border-white/25 focus:outline-none"
            />
            <button
              onClick={() => load(inputUrl)}
              disabled={!inputUrl.trim()}
              className="rounded-full bg-white px-6 py-2.5 text-sm font-medium text-neutral-950
                         transition hover:bg-neutral-200 disabled:opacity-40"
            >
              {activeUrl ? "Reload" : "Load"}
            </button>
            <button
              onClick={unload}
              disabled={!activeUrl}
              title="Tears the viewer down. Reload afterwards and watch the texture count."
              className="rounded-full bg-white/10 px-6 py-2.5 text-sm transition hover:bg-white/20 disabled:opacity-40"
            >
              Unload
            </button>
          </div>

          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((sample) => (
              <button
                key={sample.url}
                onClick={() => {
                  setInputUrl(sample.url);
                  load(sample.url);
                }}
                className={`rounded-full border px-3 py-1.5 text-xs transition ${
                  inputUrl === sample.url
                    ? "border-white/30 bg-white/15 text-neutral-100"
                    : "border-white/10 bg-white/5 text-neutral-400 hover:bg-white/10"
                }`}
              >
                {sample.label}
                <span className="ml-1.5 tabular-nums text-neutral-500">
                  {formatBytes(sample.size)}
                </span>
              </button>
            ))}
          </div>

          <p className="text-xs text-neutral-500">{sampleNote}</p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <div className="min-w-0 space-y-3">
            <div className="aspect-video w-full overflow-hidden rounded-xl border border-white/10 bg-neutral-950">
              {activeUrl ? (
                <SplatStage
                  key={cycle}
                  url={activeUrl}
                  pose={pose}
                  framing={framing}
                  onStatus={setStatus}
                  onFrameStats={handleFrameStats}
                  className="h-full w-full"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center px-6 text-center text-sm text-neutral-500">
                  Nothing loaded. Pick a sample or paste an asset URL.
                </div>
              )}
            </div>

            {/* The stage already prints the error text. This adds only what the
                error itself cannot say: what usually causes it. */}
            {status.phase === "failed" && (
              <p className="rounded-lg border border-white/10 bg-white/5 px-4 py-3 text-sm text-neutral-400">
                If the address is right, this is almost always the host refusing cross-origin
                reads or the machine being offline. Nothing is drawn in place of the asset.
              </p>
            )}

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <Metric
                label="Frames per second"
                value={frameStats ? String(frameStats.fps) : "—"}
                hint={
                  device?.refreshHz ? `display caps at ~${device.refreshHz}` : undefined
                }
              />
              <Metric
                label="Worst second"
                value={worstFps !== null ? String(worstFps) : "—"}
                hint="lowest since load"
              />
              <Metric
                label="Slowest frame"
                value={frameStats ? `${frameStats.worstFrameMs} ms` : "—"}
                hint="in the last window"
              />
              <Metric
                label="Frames rendered"
                value={frameStats ? frameStats.frames.toLocaleString() : "—"}
              />
              <Metric
                label="Splats"
                value={status.splatCount !== null ? status.splatCount.toLocaleString() : "—"}
              />
              <Metric
                label="Downloaded"
                value={status.bytesLoaded > 0 ? formatBytes(status.bytesLoaded) : "—"}
                hint={throughput ?? undefined}
              />
              <Metric
                label="Time to first frame"
                value={
                  status.timeToFirstFrameMs !== null ? `${status.timeToFirstFrameMs} ms` : "—"
                }
                hint="from load to first drawn frame"
              />
              <Metric
                label="Download / unpack"
                value={
                  status.downloadMs !== null
                    ? `${status.downloadMs} / ${status.decodeMs !== null ? status.decodeMs : "—"} ms`
                    : "—"
                }
              />
              <Metric
                label="Render resolution"
                value={
                  frameStats ? `${frameStats.drawWidth}×${frameStats.drawHeight}` : "—"
                }
                hint={frameStats ? `at ${frameStats.pixelRatio}× device pixels` : undefined}
              />
              <Metric
                label="GPU objects"
                value={
                  frameStats
                    ? `${frameStats.textures} tex / ${frameStats.geometries} geo`
                    : "—"
                }
                hint="should not climb across reloads"
              />
              <Metric label="Load cycles" value={cycle > 0 ? String(cycle) : "—"} />
              <Metric
                label="Graphics"
                value={device ? (device.supported ? "WebGL2" : "unsupported") : "—"}
                hint={device?.renderer ?? undefined}
              />
            </div>
          </div>

          <aside className="space-y-5">
            <section className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-baseline justify-between">
                <h2 className="text-sm font-medium text-neutral-200">Head pose</h2>
                <button
                  onClick={() => setPose(DEFAULT_POSE)}
                  className="text-xs text-neutral-400 underline-offset-4 hover:underline"
                >
                  Centre
                </button>
              </div>
              <p className="text-xs text-neutral-500">
                The same angles the call pipeline will drive. Moving these proves the pose reaches
                the renderer per frame, not only at load.
              </p>
              <Slider
                label="Yaw"
                unit="°"
                min={-60}
                max={60}
                value={pose.yaw}
                onChange={(yaw) => setPose((p) => ({ ...p, yaw }))}
              />
              <Slider
                label="Pitch"
                unit="°"
                min={-40}
                max={40}
                value={pose.pitch}
                onChange={(pitch) => setPose((p) => ({ ...p, pitch }))}
              />
              <Slider
                label="Roll"
                unit="°"
                min={-30}
                max={30}
                value={pose.roll}
                onChange={(roll) => setPose((p) => ({ ...p, roll }))}
              />
            </section>

            <section className="space-y-3 rounded-xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-baseline justify-between">
                <h2 className="text-sm font-medium text-neutral-200">Framing</h2>
                <button
                  onClick={() => setFraming(DEFAULT_FRAMING)}
                  className="text-xs text-neutral-400 underline-offset-4 hover:underline"
                >
                  Reset
                </button>
              </div>
              <p className="text-xs text-neutral-500">
                Distance and height are multiples of the subject&rsquo;s own size, so the same
                framing holds across assets trained at different scales.
              </p>
              <Slider
                label="Distance"
                unit="×"
                min={0.5}
                max={6}
                step={0.1}
                value={framing.distance}
                onChange={(distance) => setFraming((f) => ({ ...f, distance }))}
              />
              <Slider
                label="Height"
                unit="×"
                min={-1.5}
                max={1.5}
                step={0.05}
                value={framing.height}
                onChange={(height) => setFraming((f) => ({ ...f, height }))}
              />
              <Slider
                label="Orbit"
                unit="°"
                min={-180}
                max={180}
                value={framing.orbit}
                onChange={(orbit) => setFraming((f) => ({ ...f, orbit }))}
              />
              <Slider
                label="Field of view"
                unit="°"
                min={20}
                max={90}
                value={framing.fov}
                onChange={(fov) => setFraming((f) => ({ ...f, fov }))}
              />
            </section>

            <section className="rounded-xl border border-white/10 bg-white/5 p-4">
              <h2 className="text-sm font-medium text-neutral-200">Reading these numbers</h2>
              <ul className="mt-2 space-y-1.5 text-xs text-neutral-400">
                <li>
                  Frame rate cannot exceed the display&rsquo;s refresh rate, so compare against the
                  cap rather than against 60.
                </li>
                <li>
                  Resolution is reported because a frame rate without one is meaningless: the same
                  scene at twice the pixel ratio costs roughly four times as much.
                </li>
                <li>
                  A desktop number is an upper bound. The decision needs the same measurement taken
                  on a mid-range phone.
                </li>
              </ul>
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}
