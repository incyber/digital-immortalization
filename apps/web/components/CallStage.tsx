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
  // Undefined until we know. A machine with no webcam, or a visitor who
  // declines it, must still get a call: the camera feeds the vision channel,
  // which is an enrichment, while the microphone is the conversation itself.
  const [cameraAvailable, setCameraAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  // Probe before connecting. LiveKitRoom treats a failed track request as a
  // connection failure and disconnects, so asking for a camera that is not
  // there would end the call rather than degrade it.
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      ?.getUserMedia({ video: true })
      .then((stream) => {
        stream.getTracks().forEach((t) => t.stop());
        if (!cancelled) setCameraAvailable(true);
      })
      .catch(() => {
        if (!cancelled) setCameraAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (cameraAvailable === null) {
    return (
      <div className="flex h-full w-full items-center justify-center text-neutral-500">
        Checking your camera and microphone…
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
