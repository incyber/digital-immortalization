"use client";

// The call surface.
//
// The likeness fills the frame; the visitor's own camera sits in the corner.
// The camera is published at low resolution on purpose - nothing renders it,
// it exists only so the vision channel can sample it, and 320x240 is more
// than the vision model needs at the size it downscales to anyway.
//
// Everything here is drawn on black in both appearances. The page above wraps
// this in .always-dark, so the semantic colours below resolve to their dark
// values without this file knowing anything about themes.

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
import { Button } from "@/components/ui/Button";
import type { SessionDetails } from "@/lib/gateway";

function AvatarVideo() {
  const tracks = useTracks([Track.Source.Camera], { onlySubscribed: true });
  const participants = useRemoteParticipants();
  const agentTrack = tracks.find((t) => !t.participant.isLocal);

  if (!agentTrack) {
    return (
      <div
        className="flex h-full w-full items-center justify-center text-subhead text-label-tertiary"
        aria-live="polite"
      >
        {participants.length === 0 ? "Waiting for them to join…" : "Connecting video…"}
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
        <AvatarVideo />
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
