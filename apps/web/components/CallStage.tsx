"use client";

// The call surface.
//
// The likeness fills the frame; the visitor's own camera sits in the corner.
// The camera is published at low resolution on purpose - nothing renders it,
// it exists only so the vision channel can sample it, and 320x240 is more
// than the vision model needs at the size it downscales to anyway.

import {
  LiveKitRoom,
  RoomAudioRenderer,
  useRemoteParticipants,
  useTracks,
  VideoTrack,
} from "@livekit/components-react";
import { Track } from "livekit-client";
import "@livekit/components-styles";
import { useEffect, useState } from "react";
import type { SessionDetails } from "@/lib/api";

function AvatarVideo() {
  const tracks = useTracks([Track.Source.Camera], { onlySubscribed: true });
  const participants = useRemoteParticipants();
  const agentTrack = tracks.find((t) => !t.participant.isLocal);

  if (!agentTrack) {
    return (
      <div className="flex h-full w-full items-center justify-center text-neutral-500">
        {participants.length === 0
          ? "Waiting for the avatar to join…"
          : "Connecting video…"}
      </div>
    );
  }
  return <VideoTrack trackRef={agentTrack} className="h-full w-full object-contain" />;
}

function SelfView() {
  const tracks = useTracks([Track.Source.Camera]);
  const local = tracks.find((t) => t.participant.isLocal);
  if (!local) return null;
  return (
    <div className="absolute bottom-5 right-5 h-32 w-44 overflow-hidden rounded-lg
                    border border-white/15 bg-black/60 shadow-xl">
      <VideoTrack trackRef={local} className="h-full w-full object-cover" />
      <span className="absolute bottom-1 left-2 text-[10px] uppercase tracking-wide text-white/70">
        seen by the avatar
      </span>
    </div>
  );
}

export function CallStage({
  session,
  onLeave,
}: {
  session: SessionDetails;
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
      <div className="flex h-full w-full items-center justify-center text-neutral-500">
        Checking your camera and microphone…
      </div>
    );
  }

  // Without a microphone there is no conversation, only a video of someone
  // waiting. Say so rather than connecting to a call that cannot work.
  if (!micAvailable) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-4 px-6 text-center">
        <p className="text-neutral-300">A microphone is required for a call.</p>
        <p className="max-w-sm text-sm text-neutral-500">
          Allow microphone access for this site, then try again. The camera is
          optional — the avatar can talk without seeing you.
        </p>
        <button
          onClick={onLeave}
          className="rounded-full bg-white/10 px-6 py-2.5 text-sm text-white hover:bg-white/20"
        >
          Back
        </button>
      </div>
    );
  }

  if (failure) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-4 px-6 text-center">
        <p className="text-neutral-300">The call ended unexpectedly.</p>
        <p className="max-w-sm text-sm text-neutral-500">{failure}</p>
        <button
          onClick={onLeave}
          className="rounded-full bg-white/10 px-6 py-2.5 text-sm text-white hover:bg-white/20"
        >
          Back
        </button>
      </div>
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
      className="relative h-full w-full bg-neutral-950"
    >
      <div className="relative h-full w-full">
        <AvatarVideo />
        {cameraAvailable ? (
          <SelfView />
        ) : (
          <div className="absolute bottom-5 right-5 rounded-lg border border-white/10
                          bg-black/60 px-3 py-2 text-xs text-neutral-400">
            No camera — the avatar can hear you but cannot see you
          </div>
        )}
      </div>

      <RoomAudioRenderer />

      <div className="absolute bottom-5 left-1/2 flex -translate-x-1/2 gap-3">
        <button
          onClick={() => setMuted((m) => !m)}
          className="rounded-full bg-white/10 px-5 py-2.5 text-sm text-white
                     backdrop-blur transition hover:bg-white/20"
        >
          {muted ? "Unmute" : "Mute"}
        </button>
        <button
          onClick={onLeave}
          className="rounded-full bg-red-600 px-5 py-2.5 text-sm text-white
                     transition hover:bg-red-500"
        >
          End call
        </button>
      </div>
    </LiveKitRoom>
  );
}
