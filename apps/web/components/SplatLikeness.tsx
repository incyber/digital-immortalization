"use client";

// A built likeness, rendered on this device and moved by pose data.
//
// The split between this file and lib/pose.ts is the useful one: the wire
// format has no idea a browser exists, and this has no idea where the bytes
// came from. The call feeds it a LiveKit data channel; the internal harness
// feeds it frames it encoded a millisecond earlier. Both go through the same
// decoder, so the harness is testing the thing that ships.

import { useCallback, useEffect, useRef, useState } from "react";
import { SplatStage } from "@/components/SplatStage";
import { createPoseReceiver, type PoseReceiver } from "@/lib/pose";
import { DEFAULT_FRAMING, type CameraFraming, type HeadPose, type SplatStatus } from "@/lib/splat";

/**
 * How the camera sits for a call: square on, at head height, at the distance
 * a person sits from a laptop. Fixed rather than adjustable — a call is not a
 * model viewer, and a visitor who can orbit the likeness will orbit it round
 * to the side no photograph covered.
 */
export const CALL_FRAMING: CameraFraming = { ...DEFAULT_FRAMING };

const NEUTRAL: HeadPose = { yaw: 0, pitch: 0, roll: 0 };

export interface PoseChannel {
  /** The rotation to draw this frame. */
  pose: HeadPose;
  /** Hand it bytes off the wire. Anything unrecognisable is dropped. */
  accept: (bytes: Uint8Array) => void;
  receiver: PoseReceiver;
}

/**
 * Turns a stream of wire frames into a rotation React can render.
 *
 * The coalescing is the reason this is a hook and not three lines inline.
 * Frames arrive on the transport's schedule, which is 25 a second when the
 * network is calm and a burst of six at once when it is not; rendering each
 * one would re-render the surface six times for a single visible frame. So
 * every frame updates the receiver immediately — ordering has to be judged in
 * arrival order — and a single animation frame is scheduled to publish
 * whatever the latest state turned out to be.
 *
 * The unchanged-pose short-circuit matters as much: SplatStage pushes pose to
 * the viewer from an effect keyed on the angles, so returning the previous
 * object when nothing moved keeps a still head from doing work 25 times a
 * second for the length of a call.
 */
export function usePoseChannel(): PoseChannel {
  // State rather than a ref, and the factory passed rather than called: this
  // is the one form that builds exactly one receiver per mount without
  // reading a ref while rendering. The setter is discarded — the receiver is
  // an identity, and replacing it would mean forgetting the head's position.
  const [receiver] = useState<PoseReceiver>(createPoseReceiver);

  const [pose, setPose] = useState<HeadPose>(NEUTRAL);
  const scheduled = useRef<number | null>(null);

  const publish = useCallback(() => {
    scheduled.current = null;
    const next = receiver.headPose();
    setPose((previous) =>
      previous.yaw === next.yaw && previous.pitch === next.pitch && previous.roll === next.roll
        ? previous
        : next,
    );
  }, [receiver]);

  const accept = useCallback(
    (bytes: Uint8Array) => {
      if (!receiver.accept(bytes)) return;
      if (scheduled.current === null) scheduled.current = requestAnimationFrame(publish);
    },
    [receiver, publish],
  );

  useEffect(
    () => () => {
      if (scheduled.current !== null) cancelAnimationFrame(scheduled.current);
    },
    [],
  );

  return { pose, accept, receiver };
}

export interface SplatLikenessProps {
  url: string;
  credentials: RequestCredentials;
  /** Stable reference only; see SplatStage. */
  headers?: Record<string, string>;
  pose: HeadPose;
  /**
   * The asset cannot be drawn on this device — it failed to load, or the
   * renderer refused the hardware. The caller is expected to show the other
   * source rather than to explain this.
   */
  onUnavailable: () => void;
  className?: string;
}

export function SplatLikeness({
  url,
  credentials,
  headers,
  pose,
  onUnavailable,
  className = "",
}: SplatLikenessProps) {
  const handleStatus = useCallback(
    (status: SplatStatus) => {
      if (status.phase === "failed" || status.phase === "unsupported") onUnavailable();
    },
    [onUnavailable],
  );

  return (
    <SplatStage
      url={url}
      credentials={credentials}
      headers={headers}
      pose={pose}
      framing={CALL_FRAMING}
      onStatus={handleStatus}
      className={className}
    />
  );
}
