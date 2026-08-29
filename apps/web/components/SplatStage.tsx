"use client";

// The splat surface.
//
// One splat asset, filling whatever box it is given, with the loading state
// and the frame rate as visible outputs rather than internal details. The
// component is deliberately thin: every decision about WebGL, downloading and
// teardown lives in lib/splat.ts, so this file only has to get the React
// lifecycle right.
//
// Getting that lifecycle right is the whole job. A viewer torn down and rebuilt
// on an incidental re-render would restart a several-hundred-megabyte download
// and burn a WebGL context each time, so the effect that owns the viewer
// depends on the asset URL and nothing else, and the callbacks reach it
// through refs.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createSplatViewer,
  formatBytes,
  idleStatus,
  type CameraFraming,
  type FrameStats,
  type HeadPose,
  type SplatStatus,
  type SplatViewer,
} from "@/lib/splat";

export interface SplatStageProps {
  /** The splat asset to render: .spz, .ply, .splat, .ksplat or a SOG bundle. */
  url: string;
  /** Head orientation for this frame. Omit for a subject facing forward. */
  pose?: HeadPose;
  /** Where the camera sits relative to the subject. */
  framing?: CameraFraming;
  /** Credential mode for the asset fetch. See SplatViewerOptions. */
  credentials?: RequestCredentials;
  /**
   * Extra headers for the asset fetch. Must be a stable reference — the viewer
   * is keyed on it, so a fresh object each render would restart the download.
   */
  headers?: Record<string, string>;
  /**
   * Measured frame rate, reported roughly twice a second. This is the number
   * the device-side rendering decision turns on, so it is a first-class
   * output of the component rather than something to read off a dev tool.
   */
  onFrameStats?: (stats: FrameStats) => void;
  /** Load progress, timings and failures. */
  onStatus?: (status: SplatStatus) => void;
  /** Capped device pixel ratio; raising it costs frame rate quadratically. */
  maxPixelRatio?: number;
  className?: string;
}

export function SplatStage({
  url,
  pose,
  framing,
  credentials = "omit",
  headers,
  onFrameStats,
  onStatus,
  maxPixelRatio = 2,
  className = "",
}: SplatStageProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<SplatViewer | null>(null);
  const [status, setStatus] = useState<SplatStatus>(idleStatus);

  // Held in refs so that a parent which passes inline callbacks — the normal
  // case — does not cause the viewer below to be rebuilt on every render.
  const onFrameStatsRef = useRef(onFrameStats);
  const onStatusRef = useRef(onStatus);

  // Latest pose and framing, read once when a viewer starts. Without this a
  // reload would build the new viewer facing forward while the caller still
  // believes the head is turned, and the pose effect below would not correct
  // it because the angles themselves never changed.
  const poseRef = useRef(pose);
  const framingRef = useRef(framing);

  // Kept current after each commit rather than during render: a ref written
  // mid-render is not guaranteed to survive a render React chooses to discard.
  //
  // This effect must stay DECLARED ABOVE the viewer effect below. Effects run
  // in declaration order within a commit, which is the only reason the refs
  // hold current values when a URL change builds a replacement viewer. Move it
  // below and a reload silently starts the new viewer from a stale pose.
  useEffect(() => {
    onFrameStatsRef.current = onFrameStats;
    onStatusRef.current = onStatus;
    poseRef.current = pose;
    framingRef.current = framing;
  });

  const handleStatus = useCallback((next: SplatStatus) => {
    setStatus(next);
    onStatusRef.current?.(next);
  }, []);

  const handleFrameStats = useCallback((next: FrameStats) => {
    onFrameStatsRef.current?.(next);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !url) return;

    setStatus(idleStatus());
    const viewer = createSplatViewer({
      container,
      url,
      pose: poseRef.current,
      framing: framingRef.current,
      maxPixelRatio,
      credentials,
      headers,
      onStatus: handleStatus,
      onFrameStats: handleFrameStats,
    });
    viewerRef.current = viewer;

    return () => {
      viewerRef.current = null;
      viewer.dispose();
    };
  }, [url, maxPixelRatio, credentials, headers, handleStatus, handleFrameStats]);

  // Pose and framing are pushed imperatively. Routing them through the effect
  // that owns the viewer would tie a slider drag to a full reload. These stay
  // declared below the viewer effect so viewerRef is populated when they run.
  const { yaw = 0, pitch = 0, roll = 0 } = pose ?? {};
  useEffect(() => {
    viewerRef.current?.setPose({ yaw, pitch, roll });
  }, [yaw, pitch, roll]);

  const { distance, height, orbit, fov } = framing ?? {};
  useEffect(() => {
    if (distance === undefined || height === undefined || orbit === undefined || fov === undefined) {
      return;
    }
    viewerRef.current?.setFraming({ distance, height, orbit, fov });
  }, [distance, height, orbit, fov]);

  const loading = status.phase === "downloading" || status.phase === "decoding";

  return (
    <div className={`always-dark relative h-full w-full overflow-hidden bg-surface ${className}`}>
      {/* The viewer appends its own canvas here and removes it on teardown. */}
      <div ref={containerRef} className="absolute inset-0" />

      {status.phase === "unsupported" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
          <p className="text-body text-label">This device cannot render the likeness.</p>
          <p className="max-w-sm text-subhead text-label-secondary">{status.message}</p>
        </div>
      )}

      {status.phase === "failed" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
          <p className="text-body text-label">The likeness could not be loaded.</p>
          <p className="max-w-md text-subhead text-red">{status.message}</p>
        </div>
      )}

      {loading && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6"
          aria-live="polite"
        >
          <p className="text-subhead text-label">
            {status.phase === "decoding"
              ? "Unpacking the likeness…"
              : status.percent !== null
                ? `Loading the likeness — ${status.percent}%`
                : "Loading the likeness…"}
          </p>

          <div
            role="progressbar"
            aria-label="Loading the likeness"
            aria-valuemin={0}
            aria-valuemax={100}
            {...(status.percent !== null ? { "aria-valuenow": status.percent } : {})}
            className="h-1 w-56 max-w-[70vw] overflow-hidden rounded-full bg-fill"
          >
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
              style={{
                // With no Content-Length there is no honest percentage, so the
                // bar sits at a fixed sliver rather than inventing a position.
                width: status.percent !== null ? `${Math.max(2, status.percent)}%` : "12%",
              }}
            />
          </div>

          <p className="text-footnote tabular-nums text-label-tertiary">
            {status.bytesTotal !== null
              ? `${formatBytes(status.bytesLoaded)} of ${formatBytes(status.bytesTotal)}`
              : `${formatBytes(status.bytesLoaded)} — total size not reported by the server`}
          </p>
        </div>
      )}
    </div>
  );
}
