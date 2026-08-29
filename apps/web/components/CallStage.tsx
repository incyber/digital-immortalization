"use client";

// The call surface.
//
// The likeness fills the frame; the visitor's own camera sits in the corner.
// The camera is published at low resolution on purpose - nothing renders it,
// it exists only so the vision channel can sample it, and 320x240 is more
// than the vision model needs at the size it downscales to anyway.
//
// Two things can fill that frame. A built Gaussian splat drawn by this
// device and turned by pose frames off the data channel, or the video track
// the server renders and publishes. Which one was settled before the room
// connected - see lib/likeness.ts - and arrives here already decided, so
// nothing swaps under somebody mid-sentence. The one exception moves in a
// single direction and only on failure: a splat that cannot be loaded gives
// way to the video track, without a word about it on screen.
//
// The voice is never behind any of this. RoomAudioRenderer sits outside the
// surface below and is mounted the moment the room is, so a likeness still
// downloading delays pixels and nothing else. Audio is the person; the
// picture is the accompaniment.
//
// Everything here is drawn on black in both appearances. The page above wraps
// this in .always-dark, so the semantic colours below resolve to their dark
// values without this file knowing anything about themes.

import {
  LiveKitRoom,
  RoomAudioRenderer,
  useRemoteParticipants,
  useRoomContext,
  useTracks,
  VideoTrack,
} from "@livekit/components-react";
import { RoomEvent, Track } from "livekit-client";
import type { DataPacket_Kind, RemoteParticipant } from "livekit-client";
import "@livekit/components-styles";
import { useCallback, useEffect, useState } from "react";
import { SplatLikeness, usePoseChannel } from "@/components/SplatLikeness";
import { Button } from "@/components/ui/Button";
import type { SessionDetails, SplatAsset } from "@/lib/gateway";
import type { LikenessPlan } from "@/lib/likeness";
import { POSE_TOPIC } from "@/lib/pose";

function AvatarVideo({ noLikeness = false }: { noLikeness?: boolean }) {
  const tracks = useTracks([Track.Source.Camera], { onlySubscribed: true });
  const participants = useRemoteParticipants();
  const agentTrack = tracks.find((t) => !t.participant.isLocal);

  if (!agentTrack) {
    return (
      <div
        className="flex h-full w-full items-center justify-center px-6 text-center
                   text-subhead text-label-tertiary"
        aria-live="polite"
      >
        {/* Both sources gone is the one state that has to be said out loud.
            Waiting and connecting are moments; this one does not resolve, and
            leaving a black rectangle to stand for it would have somebody
            sitting in front of it wondering whether it is their end. */}
        {noLikeness
          ? "You can hear them, but there is no picture on this call."
          : participants.length === 0
            ? "Waiting for them to join…"
            : "Connecting video…"}
      </div>
    );
  }
  return <VideoTrack trackRef={agentTrack} className="h-full w-full object-contain" />;
}

/**
 * The splat, turned by whatever the far end sends.
 *
 * Subscribed straight to the room rather than through useDataChannel, which
 * keeps the last message in state: at 25 frames a second that would re-render
 * this subtree on the transport's schedule instead of the display's. The
 * frames go to usePoseChannel, which coalesces them onto animation frames.
 */
function PoseDrivenLikeness({
  asset,
  onUnavailable,
}: {
  asset: SplatAsset;
  onUnavailable: () => void;
}) {
  const room = useRoomContext();
  const { pose, accept } = usePoseChannel();

  useEffect(() => {
    const receive = (
      payload: Uint8Array,
      _participant?: RemoteParticipant,
      _kind?: DataPacket_Kind,
      topic?: string,
    ) => {
      // A packet on somebody else's topic is not ours to read. One with no
      // topic at all is still offered to the decoder, which rejects anything
      // that is not a pose frame on length and header before it can move a
      // head - a sender that forgets to set the topic should still work.
      if (topic !== undefined && topic !== POSE_TOPIC) return;
      accept(payload);
    };

    room.on(RoomEvent.DataReceived, receive);
    return () => {
      room.off(RoomEvent.DataReceived, receive);
    };
  }, [room, accept]);

  return (
    <SplatLikeness
      url={asset.url}
      credentials={asset.credentials}
      headers={asset.headers}
      pose={pose}
      onUnavailable={onUnavailable}
    />
  );
}

/**
 * Whichever source was chosen, with one silent retreat available.
 *
 * The video track stays subscribed even while the splat is drawing. It costs
 * bandwidth that the whole point of this was to save, and it is worth it: it
 * is the only thing that makes the retreat below instant rather than a
 * reconnection in front of somebody.
 */
function AvatarSurface({ likeness }: { likeness: LikenessPlan }) {
  const [splatUnavailable, setSplatUnavailable] = useState(false);
  const onUnavailable = useCallback(() => setSplatUnavailable(true), []);

  if (likeness.renderer === "splat" && likeness.asset && !splatUnavailable) {
    return <PoseDrivenLikeness asset={likeness.asset} onUnavailable={onUnavailable} />;
  }
  return <AvatarVideo noLikeness={splatUnavailable} />;
}

function SelfView() {
  const tracks = useTracks([Track.Source.Camera]);
  const local = tracks.find((t) => t.participant.isLocal);
  if (!local) return null;
  return (
    <div
      className="absolute right-6 bottom-6 h-32 w-44 overflow-hidden rounded-2xl
                 border border-separator bg-black/60"
    >
      <VideoTrack trackRef={local} className="h-full w-full object-cover" />
      <span className="absolute bottom-2 left-3 text-caption text-label-secondary">
        seen by them
      </span>
    </div>
  );
}

/** Everything that is not a call: a sentence, and a way back. */
function Stopped({ title, detail, onLeave }: { title: string; detail: string; onLeave: () => void }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-6 px-6 text-center">
      <p className="text-body text-label">{title}</p>
      <p className="max-w-sm text-subhead text-label-secondary">{detail}</p>
      <Button rank="grey" onClick={onLeave}>
        Back
      </Button>
    </div>
  );
}

export function CallStage({
  session,
  likeness,
  onLeave,
}: {
  session: SessionDetails;
  /** Settled before this component mounted, and not re-read afterwards. */
  likeness: LikenessPlan;
  onLeave: () => void;
}) {
  const [muted, setMuted] = useState(false);
  // Undefined until probed. A machine with no webcam, or a visitor who
  // declines it, must still get a call: the camera feeds the vision channel,
  // which is an enrichment, while the microphone is the conversation itself.
  const [cameraAvailable, setCameraAvailable] = useState<boolean | null>(null);
  const [micAvailable, setMicAvailable] = useState<boolean | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  // Probe each device before connecting. LiveKitRoom treats a failed track
  // request as a connection failure and disconnects the whole room, so asking
  // for hardware that is absent would end the call instead of degrading it.
  useEffect(() => {
    let cancelled = false;

    async function probe(constraints: MediaStreamConstraints) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        stream.getTracks().forEach((t) => t.stop());
        return true;
      } catch {
        return false;
      }
    }

    (async () => {
      const [video, audio] = await Promise.all([
        probe({ video: true }),
        probe({ audio: true }),
      ]);
      if (cancelled) return;
      setCameraAvailable(video);
      setMicAvailable(audio);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  if (cameraAvailable === null || micAvailable === null) {
    return (
      <div
        className="flex h-full w-full items-center justify-center text-subhead text-label-tertiary"
        aria-live="polite"
      >
        Checking your camera and microphone…
      </div>
    );
  }

  // Without a microphone there is no conversation, only a video of someone
  // waiting. Say so rather than connecting to a call that cannot work.
  if (!micAvailable) {
    return (
      <Stopped
        title="A microphone is needed for a call."
        detail="Allow microphone access for this site, then try again. The camera is optional — they can talk without seeing you."
        onLeave={onLeave}
      />
    );
  }

  if (failure) {
    return (
      <Stopped title="The call ended unexpectedly." detail={failure} onLeave={onLeave} />
    );
  }

  return (
    <LiveKitRoom
      serverUrl={session.url}
      token={session.token}
      connect
      audio={!muted}
      video={cameraAvailable}
      onDisconnected={onLeave}
      onError={(e) => setFailure(e.message)}
      className="relative h-full w-full bg-surface"
    >
      <div className="relative h-full w-full">
        <AvatarSurface likeness={likeness} />
        {cameraAvailable ? (
          <SelfView />
        ) : (
          <div
            className="absolute right-6 bottom-6 rounded-xl border border-separator bg-black/60
                       px-4 py-2 text-footnote text-label-secondary"
          >
            No camera — they can hear you but cannot see you
          </div>
        )}
      </div>

      {/* Outside the surface above, and never gated on it. Their voice must
          start the moment the room is up, whether or not a likeness is still
          arriving over the wire. */}
      <RoomAudioRenderer />

      <div className="absolute bottom-6 left-1/2 flex -translate-x-1/2 gap-3">
        <Button rank="grey" onClick={() => setMuted((m) => !m)}>
          {muted ? "Unmute" : "Mute"}
        </Button>
        <Button rank="destructive" onClick={onLeave}>
          End call
        </Button>
      </div>
    </LiveKitRoom>
  );
}
