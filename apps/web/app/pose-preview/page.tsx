"use client";

// Internal instrument — not a customer surface.
//
// One question: do the bytes on the wire move the head? A call cannot answer
// it, because a call needs a room, a far end and a person who has died. So
// this page plays the far end. It builds pose frames, encodes them with the
// same encoder the specification in lib/pose.ts is written from, and hands
// the raw bytes to the same receiver the call uses. Nothing is short-cut: if
// the format and the decoder ever disagree, the head here stops moving.
//
// The fault switches are the point of it. Real networks lose frames, deliver
// them out of order and hand over the occasional packet that is not what it
// claims to be, and none of those may take a call down or jerk a face about.
// Turning them all on and watching the head keep moving smoothly, while the
// counters climb, is the test.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SplatLikeness, usePoseChannel } from "@/components/SplatLikeness";
import { Button } from "@/components/ui/Button";
import { Notice } from "@/components/ui/Notice";
import { Screen } from "@/components/ui/Screen";
import {
  encodePoseFrame,
  LIVE_CHANNELS,
  POSE_CHANNELS,
  POSE_CHANNEL_COUNT,
  POSE_FRAME_BYTES,
  POSE_VISEMES,
  POSE_VISEME_COUNT,
  type PoseTally,
} from "@/lib/pose";

// A public sample from the renderer's authors, served with permissive CORS.
// Not a face, and not the size of a real capture — neither matters here,
// because what is being measured is whether it turns when told to.
const WORKING_ASSET = "https://sparkjs.dev/assets/splats/robot-head.spz";

// Same host, no such file. Proves the surface reports an asset it cannot
// load rather than sitting on a black rectangle. In the call this is the
// point where the video track quietly takes over instead.
const BROKEN_ASSET = "https://sparkjs.dev/assets/splats/no-such-likeness.spz";

const FRAME_MS = 40; // 25 fps

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2">
      <span className="text-footnote text-label-secondary">{label}</span>
      <span className="text-footnote tabular-nums text-label">{value}</span>
    </div>
  );
}

function Toggle({
  label,
  detail,
  on,
  onChange,
}: {
  label: string;
  detail: string;
  on: boolean;
  onChange: (on: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-2">
      <input
        type="checkbox"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 size-4 shrink-0 accent-[var(--accent)]"
      />
      <span>
        <span className="block text-subhead text-label">{label}</span>
        <span className="block text-footnote text-label-secondary">{detail}</span>
      </span>
    </label>
  );
}

export default function PosePreview() {
  const [assetUrl, setAssetUrl] = useState(WORKING_ASSET);
  const [unavailable, setUnavailable] = useState(false);

  const [sending, setSending] = useState(true);
  const [losing, setLosing] = useState(false);
  const [reordering, setReordering] = useState(false);
  const [corrupting, setCorrupting] = useState(false);

  const { pose, accept, receiver } = usePoseChannel();
  const [tally, setTally] = useState<PoseTally>({
    applied: 0,
    outOfOrder: 0,
    malformed: 0,
    missing: 0,
  });

  // Read on a timer rather than published from the receiver: the counters are
  // a readout, and having them drive React at 25 Hz would change the thing
  // being measured.
  useEffect(() => {
    const id = setInterval(() => setTally(receiver.tally()), 250);
    return () => clearInterval(id);
  }, [receiver]);

  // Held in a ref so flipping a switch does not restart the sender and reset
  // the sweep, which would make the head jump for reasons the network had
  // nothing to do with.
  const faults = useRef({ losing, reordering, corrupting });
  useEffect(() => {
    faults.current = { losing, reordering, corrupting };
  }, [losing, reordering, corrupting]);

  useEffect(() => {
    if (!sending) return;

    // Deliberately not zero. Sequence numbers may start anywhere and wrap, so
    // starting near the top exercises the wrap within a minute of watching.
    let sequence = 0xffffff00;
    const startedAt = performance.now();

    const id = setInterval(() => {
      const t = (performance.now() - startedAt) / 1000;

      const channels = new Float32Array(POSE_CHANNEL_COUNT);
      // Three incommensurable periods, so the head never repeats a position
      // and a frozen render is obvious rather than plausible.
      channels[0] = 32 * Math.sin(t * 0.83); // head_yaw
      channels[1] = 14 * Math.sin(t * 0.51 + 1.1); // head_pitch
      channels[2] = 9 * Math.sin(t * 0.37 + 2.3); // head_roll
      // Everything below is sent, checked and dropped. It is populated with
      // real-looking values on purpose: if any of it ever leaked into the
      // render, the likeness would visibly twitch here.
      channels[3] = 6 * Math.sin(t * 1.7);
      channels[4] = 4 * Math.sin(t * 1.3);
      const blink = t % 3.4 < 0.14 ? 1 : 0;
      channels[5] = blink;
      channels[6] = blink;
      channels[14] = Math.max(0, Math.sin(t * 5.5)); // jaw_open
      channels[15] = 0.3;
      channels[16] = 0.3;
      channels[18] = 3 * Math.sin(t * 0.21); // torso_lean
      channels[19] = 0.5 + 0.5 * Math.sin(t * 0.9); // breath

      const visemes = new Float32Array(POSE_VISEME_COUNT);
      visemes[10] = Math.max(0, Math.sin(t * 5.5)); // "aa"
      visemes[0] = 1 - visemes[10]; // silence

      const bytes = encodePoseFrame({ sequence, channels, visemes });
      sequence = (sequence + 1) >>> 0;

      const { losing: lose, reordering: reorder, corrupting: corrupt } = faults.current;

      // Lost: the sequence number is still spent, which is what lets the
      // receiver count the gap rather than never learning of it.
      if (lose && Math.random() < 0.25) return;

      if (corrupt && Math.random() < 0.12) {
        const damaged = bytes.slice();
        damaged[1] = 0x09; // a version this build does not speak
        accept(damaged);
        return;
      }

      if (reorder && Math.random() < 0.2) {
        // Three frames late, so it arrives behind ones already applied.
        setTimeout(() => accept(bytes), FRAME_MS * 3);
        return;
      }

      accept(bytes);
    }, FRAME_MS);

    return () => clearInterval(id);
  }, [sending, accept]);

  const onUnavailable = useCallback(() => setUnavailable(true), []);

  const dropped = useMemo(
    () => POSE_CHANNELS.filter((_, i) => !LIVE_CHANNELS.includes(i)),
    [],
  );

  return (
    <Screen
      title="Pose over the wire"
      measure="wide"
      back={{ href: "/", label: "Home" }}
      lede={`Plays the far end of a call: ${POSE_FRAME_BYTES} bytes a frame at 25 per second, encoded and decoded exactly as the format specifies. The head below is turned by nothing else.`}
    >
      <Notice tone="attention" className="mb-8">
        Internal instrument. Not a page any customer reaches.
      </Notice>

      <div className="group-surface aspect-video w-full overflow-hidden">
        <SplatLikeness
          key={assetUrl}
          url={assetUrl}
          credentials="omit"
          pose={pose}
          onUnavailable={onUnavailable}
        />
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button
          rank={assetUrl === WORKING_ASSET ? "filled" : "grey"}
          small
          onClick={() => {
            setUnavailable(false);
            setAssetUrl(WORKING_ASSET);
          }}
        >
          Loadable asset
        </Button>
        <Button
          rank={assetUrl === BROKEN_ASSET ? "filled" : "grey"}
          small
          onClick={() => {
            setUnavailable(false);
            setAssetUrl(BROKEN_ASSET);
          }}
        >
          Asset that will not load
        </Button>
        <Button rank="grey" small onClick={() => setSending((s) => !s)}>
          {sending ? "Stop the sender" : "Start the sender"}
        </Button>
      </div>

      {unavailable && (
        <Notice tone="problem" role="status" className="mt-6">
          The surface reported that it cannot draw this asset. In a call this is the moment the
          video track takes over, with nothing said about it on screen.
        </Notice>
      )}

      {!sending && (
        <Notice tone="quiet" role="status" className="mt-6">
          The sender is stopped and the head is holding its last position. It does not ease back to
          centre: returning to neutral would be a movement, and no frame asked for one.
        </Notice>
      )}

      <div className="mt-8 grid gap-8 sm:grid-cols-2">
        <section>
          <h2 className="text-headline text-label">The network, made worse</h2>
          <div className="mt-2 divide-y divide-separator">
            <Toggle
              label="Lose a quarter of the frames"
              detail="Never delivered. The gap is counted, and the head keeps moving."
              on={losing}
              onChange={setLosing}
            />
            <Toggle
              label="Deliver some frames three late"
              detail="They arrive behind newer ones and are refused rather than applied."
              on={reordering}
              onChange={setReordering}
            />
            <Toggle
              label="Corrupt the version byte on some"
              detail="Dropped whole. Never patched up with a neutral pose."
              on={corrupting}
              onChange={setCorrupting}
            />
          </div>
        </section>

        <section>
          <h2 className="text-headline text-label">What arrived</h2>
          <div className="mt-2 divide-y divide-separator">
            <Row label="Applied" value={tally.applied.toLocaleString()} />
            <Row label="Out of order or repeated" value={tally.outOfOrder.toLocaleString()} />
            <Row label="Malformed" value={tally.malformed.toLocaleString()} />
            <Row label="Lost in transit" value={tally.missing.toLocaleString()} />
            <Row label="Yaw" value={`${pose.yaw.toFixed(1)}°`} />
            <Row label="Pitch" value={`${pose.pitch.toFixed(1)}°`} />
            <Row label="Roll" value={`${pose.roll.toFixed(1)}°`} />
          </div>
        </section>
      </div>

      <section className="mt-8">
        <h2 className="text-headline text-label">What the frame drives</h2>
        {/* Read from the same constants the renderer reads, so this cannot
            describe a face doing more than it is doing. */}
        <p className="mt-2 text-subhead text-label-secondary">
          Live — {LIVE_CHANNELS.map((i) => POSE_CHANNELS[i]).join(", ")}. These rotate the
          likeness, which has no skeleton in it, so the whole cloud turns together.
        </p>
        <p className="mt-2 text-subhead text-label-secondary">
          Received and dropped — {dropped.join(", ")}, and all {POSE_VISEME_COUNT} visemes (
          {POSE_VISEMES.join(", ")}). Nothing on this list moves anything. A static splat has no
          eyelid to close, and inventing one would be worse than the stillness.
        </p>
      </section>
    </Screen>
  );
}
