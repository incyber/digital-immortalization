// The pose wire format, and what a browser is allowed to do with it.
//
// A call renders the likeness on the visitor's own device, so the only thing
// crossing the network is the motion: 35 numbers, 25 times a second, about
// 3.7 KB/s. That is the entire economic argument — no server GPU, no encoder,
// no video bitrate — and it only holds if this file and the sender agree on
// the bytes exactly, so the format is specified here rather than discovered.
//
// The receiver is deliberately unforgiving. A frame that is the wrong length,
// the wrong version, or carries a value that is not a finite number is
// dropped whole. It is never patched up with zeroes, because zero is a
// legitimate pose — a person facing straight ahead — and substituting it
// would turn a corrupt packet into a head movement nobody made.
//
// ===========================================================================
// FORMAT "pose.v1" — implement against this, byte for byte
// ===========================================================================
//
// Transport : LiveKit data channel, topic "pose", lossy delivery.
//             Lossy is correct: a pose frame that arrives late is worthless,
//             and retransmitting it would delay the frames behind it.
// Rate      : 25 frames per second (40 ms).
// Byte order: little-endian, everywhere, including the header.
// Length    : exactly 148 bytes. Any other length is dropped.
//
//   offset  type         field
//   ------  -----------  --------------------------------------------------
//        0  uint8        magic, 0x50
//        1  uint8        version, 0x01
//        2  uint8        channel count, must be 20
//        3  uint8        viseme count, must be 15
//        4  uint32       sequence number
//        8  float32[20]  continuous channels, order below
//       88  float32[15]  viseme weights, order below
//      148  ---          end
//
// The header is 8 bytes so that both float arrays begin on a 4-byte boundary
// and can be read as typed-array views with no copy.
//
// The sender, in full:
//
//     import struct
//
//     POSE_MAGIC = 0x50
//     POSE_VERSION = 1
//     POSE_FRAME = struct.Struct("<BBBBI20f15f")   # 148 bytes
//
//     def encode_pose(sequence: int, channels, visemes) -> bytes:
//         assert len(channels) == 20 and len(visemes) == 15
//         return POSE_FRAME.pack(
//             POSE_MAGIC, POSE_VERSION, 20, 15,
//             sequence & 0xFFFFFFFF,
//             *channels, *visemes,
//         )
//
//     # await room.local_participant.publish_data(
//     #     encode_pose(seq, channels, visemes),
//     #     topic="pose",
//     #     reliable=False,
//     # )
//
// Sequence numbers start anywhere, increment by exactly one per frame sent,
// and wrap modulo 2**32. The receiver compares them as a signed 32-bit
// difference, so the wrap is not a special case for the sender to handle. A
// gap in the sequence is a lost frame and is expected; the receiver counts
// them and carries on.
//
// Angles are degrees. Weights are 0..1. Both are documented per channel and
// neither is normalised or rescaled on arrival — a sender that ships radians
// will produce a likeness that spins, which is the correct failure: visible,
// immediate, and traceable to the sender.
//
// ===========================================================================

import type { HeadPose } from "@/lib/splat";

/** LiveKit data-channel topic. Frames on other topics are not pose frames. */
export const POSE_TOPIC = "pose";

export const POSE_MAGIC = 0x50;
export const POSE_VERSION = 1;

export const POSE_CHANNEL_COUNT = 20;
export const POSE_VISEME_COUNT = 15;

const HEADER_BYTES = 8;
const CHANNELS_OFFSET = HEADER_BYTES;
const VISEMES_OFFSET = CHANNELS_OFFSET + POSE_CHANNEL_COUNT * 4;

/** Exactly 148. A frame of any other size is not a v1 frame. */
export const POSE_FRAME_BYTES = VISEMES_OFFSET + POSE_VISEME_COUNT * 4;

/**
 * The 20 continuous channels, in wire order.
 *
 * Index is the contract; the names are for people. Three of these reach the
 * renderer — see LIVE_CHANNELS below — and the rest are received, validated
 * and dropped. They are in the format anyway because the sender is producing
 * them for a rig that will eventually consume them, and a format that grew a
 * channel per quarter would break every sender each time.
 */
export const POSE_CHANNELS = [
  /*  0 */ "head_yaw", // degrees, + turns the face toward the viewer's right
  /*  1 */ "head_pitch", // degrees, + raises the chin
  /*  2 */ "head_roll", // degrees, + tips the crown toward the viewer's right
  /*  3 */ "gaze_yaw", // degrees, eyes relative to the head
  /*  4 */ "gaze_pitch", // degrees, eyes relative to the head
  /*  5 */ "blink", // 0 open .. 1 shut
  /*  6 */ "lid_upper_l", // -1 .. 1
  /*  7 */ "lid_upper_r", // -1 .. 1
  /*  8 */ "brow_inner_l", // -1 .. 1
  /*  9 */ "brow_inner_r", // -1 .. 1
  /* 10 */ "brow_outer_l", // -1 .. 1
  /* 11 */ "brow_outer_r", // -1 .. 1
  /* 12 */ "jaw_open", // 0 .. 1
  /* 13 */ "mouth_smile_l", // -1 .. 1
  /* 14 */ "mouth_smile_r", // -1 .. 1
  /* 15 */ "mouth_press", // 0 .. 1
  /* 16 */ "torso_lean", // -1 .. 1
  /* 17 */ "torso_yaw", // -0.4 .. 0.4
  /* 18 */ "shoulder_raise", // 0 .. 1
  /* 19 */ "breath", // 0 .. 1
] as const;

/**
 * The 15 viseme weights, in wire order. The set is the fifteen-mouth-shape
 * one every off-the-shelf lip-sync model emits, so a sender does not have to
 * remap; "sil" is silence and sits at index 0 for the same reason.
 */
export const POSE_VISEMES = [
  /*  0 */ "sil",
  /*  1 */ "PP",
  /*  2 */ "FF",
  /*  3 */ "TH",
  /*  4 */ "DD",
  /*  5 */ "kk",
  /*  6 */ "CH",
  /*  7 */ "SS",
  /*  8 */ "nn",
  /*  9 */ "RR",
  /* 10 */ "aa",
  /* 11 */ "E",
  /* 12 */ "ih",
  /* 13 */ "oh",
  /* 14 */ "ou",
] as const;

export const CHANNEL_HEAD_YAW = 0;
export const CHANNEL_HEAD_PITCH = 1;
export const CHANNEL_HEAD_ROLL = 2;

/**
 * WHAT ACTUALLY MOVES, AND WHAT DOES NOT.
 *
 * LIVE — channels 0, 1 and 2. Head yaw, pitch and roll become a rotation of
 * the splat. A Gaussian splat built from photographs is one rigid cloud with
 * no skeleton in it, so "the head turns" is drawn as the whole likeness
 * turning. That is a real response to real data and it is what the visitor
 * sees; it is not a neck.
 *
 * DROPPED — the other 17 channels and all 15 visemes. There is nothing in a
 * static splat for them to drive. They are read, checked and discarded.
 *
 * They are dropped rather than approximated on purpose, and the temptation
 * they exist to refuse is specific: blink could be faked by squashing the
 * cloud, jaw_open by scaling it, breath by a slow pulse. Each of those is a
 * movement the person never made, shown to somebody who is looking at their
 * mother's face for evidence that she is still there. A likeness that plainly
 * does not blink is honest. One that appears to blink is a lie with a very
 * small budget.
 *
 * When a rig with per-region deformation lands, channels move from DROPPED to
 * LIVE here, and this comment is the thing that has to change with them.
 */
export const LIVE_CHANNELS: readonly number[] = [
  CHANNEL_HEAD_YAW,
  CHANNEL_HEAD_PITCH,
  CHANNEL_HEAD_ROLL,
];

/**
 * Rotation limits, in degrees.
 *
 * A splat built from photographs has detail only where a camera looked, so
 * past roughly these angles the viewer is shown the unreconstructed back of
 * somebody's head. The clamp is also what stops one corrupt-but-finite frame
 * from spinning the likeness. Senders are expected to stay inside these; the
 * clamp is a floor on how bad it can look, not a licence to send anything.
 */
export const YAW_LIMIT = 45;
export const PITCH_LIMIT = 30;
export const ROLL_LIMIT = 25;

export interface PoseFrame {
  sequence: number;
  /** 20 values, indexed by POSE_CHANNELS. */
  channels: Float32Array;
  /** 15 values, indexed by POSE_VISEMES. */
  visemes: Float32Array;
}

function clamp(value: number, limit: number): number {
  return value < -limit ? -limit : value > limit ? limit : value;
}

/** The head rotation a frame asks for, clamped to what the asset can show. */
export function headPoseOf(frame: PoseFrame): HeadPose {
  return {
    yaw: clamp(frame.channels[CHANNEL_HEAD_YAW], YAW_LIMIT),
    pitch: clamp(frame.channels[CHANNEL_HEAD_PITCH], PITCH_LIMIT),
    roll: clamp(frame.channels[CHANNEL_HEAD_ROLL], ROLL_LIMIT),
  };
}

/**
 * Decodes one frame, or returns null if these bytes are not one.
 *
 * Returning null rather than throwing because malformed frames are an
 * expected condition on a lossy channel shared with other traffic, and a
 * throw on the data-channel callback would take the call down with it.
 */
export function decodePoseFrame(bytes: Uint8Array): PoseFrame | null {
  if (bytes.byteLength !== POSE_FRAME_BYTES) return null;

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint8(0) !== POSE_MAGIC) return null;
  if (view.getUint8(1) !== POSE_VERSION) return null;
  if (view.getUint8(2) !== POSE_CHANNEL_COUNT) return null;
  if (view.getUint8(3) !== POSE_VISEME_COUNT) return null;

  const sequence = view.getUint32(4, true);

  // Copied rather than viewed over the incoming buffer. Two reasons, and the
  // second is the one that bites: the transport is free to reuse its buffer
  // after the callback returns, and a typed-array view requires the byte
  // offset to be 4-aligned, which a pooled buffer does not guarantee.
  const channels = new Float32Array(POSE_CHANNEL_COUNT);
  for (let i = 0; i < POSE_CHANNEL_COUNT; i += 1) {
    const value = view.getFloat32(CHANNELS_OFFSET + i * 4, true);
    if (!Number.isFinite(value)) return null;
    channels[i] = value;
  }

  const visemes = new Float32Array(POSE_VISEME_COUNT);
  for (let i = 0; i < POSE_VISEME_COUNT; i += 1) {
    const value = view.getFloat32(VISEMES_OFFSET + i * 4, true);
    if (!Number.isFinite(value)) return null;
    visemes[i] = value;
  }

  return { sequence, channels, visemes };
}

/**
 * Encodes one frame. The browser never sends pose — this is the executable
 * half of the specification above, used by the internal harness to drive the
 * real decoder with real bytes. If it and decodePoseFrame ever disagree, the
 * harness stops moving, which is a louder failure than a comment going stale.
 */
export function encodePoseFrame(frame: PoseFrame): Uint8Array {
  if (frame.channels.length !== POSE_CHANNEL_COUNT) {
    throw new Error(`a pose frame carries ${POSE_CHANNEL_COUNT} channels`);
  }
  if (frame.visemes.length !== POSE_VISEME_COUNT) {
    throw new Error(`a pose frame carries ${POSE_VISEME_COUNT} visemes`);
  }

  const bytes = new Uint8Array(POSE_FRAME_BYTES);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, POSE_MAGIC);
  view.setUint8(1, POSE_VERSION);
  view.setUint8(2, POSE_CHANNEL_COUNT);
  view.setUint8(3, POSE_VISEME_COUNT);
  view.setUint32(4, frame.sequence >>> 0, true);

  for (let i = 0; i < POSE_CHANNEL_COUNT; i += 1) {
    view.setFloat32(CHANNELS_OFFSET + i * 4, frame.channels[i], true);
  }
  for (let i = 0; i < POSE_VISEME_COUNT; i += 1) {
    view.setFloat32(VISEMES_OFFSET + i * 4, frame.visemes[i], true);
  }
  return bytes;
}

/** Counts kept so a call that looks wrong can be told apart from one that is. */
export interface PoseTally {
  /** Frames decoded and newer than the last one applied. */
  applied: number;
  /** Frames that arrived after a newer one had already been applied. */
  outOfOrder: number;
  /** Frames the wrong length, version or shape, or carrying NaN. */
  malformed: number;
  /** Gaps in the sequence: sent by the far end, never delivered. */
  missing: number;
}

export interface PoseReceiver {
  /** True when this frame moved the head. */
  accept(bytes: Uint8Array): boolean;
  /** The most recent accepted rotation. Neutral until the first frame lands. */
  headPose(): HeadPose;
  /** The most recent accepted frame, whole. Null until the first one lands. */
  frame(): PoseFrame | null;
  tally(): PoseTally;
}

/**
 * Signed 32-bit difference, so a sequence wrapping past 2**32 reads as +1
 * rather than as four billion frames of reordering. `| 0` is what performs
 * the wrap: it is ToInt32, not a rounding convenience.
 */
function sequenceDelta(next: number, previous: number): number {
  return (next - previous) | 0;
}

/**
 * Accumulates frames from the wire into one current head rotation.
 *
 * The ordering rule is the whole of it: a frame is applied only if it is
 * newer than the last one applied. Nothing is buffered, interpolated or held
 * back, because a pose frame is only ever wanted at the moment it arrives —
 * queuing them to smooth the motion would trade the thing the visitor is
 * actually judging, which is whether the head turns when the voice does, for
 * a smoothness nobody asked for.
 *
 * When the sender stops, the last pose stays. It does not ease back to
 * centre: returning to neutral is a movement, and no frame asked for it.
 */
export function createPoseReceiver(): PoseReceiver {
  let latest: PoseFrame | null = null;
  let pose: HeadPose = { yaw: 0, pitch: 0, roll: 0 };
  const tally: PoseTally = { applied: 0, outOfOrder: 0, malformed: 0, missing: 0 };

  return {
    accept(bytes: Uint8Array): boolean {
      const frame = decodePoseFrame(bytes);
      if (frame === null) {
        tally.malformed += 1;
        return false;
      }

      if (latest !== null) {
        const delta = sequenceDelta(frame.sequence, latest.sequence);
        // Zero is a duplicate, negative is a reordering. Neither is an error
        // on a lossy channel and neither is worth applying: the head is
        // already at least this current.
        if (delta <= 0) {
          tally.outOfOrder += 1;
          return false;
        }
        if (delta > 1) tally.missing += delta - 1;
      }

      latest = frame;
      pose = headPoseOf(frame);
      tally.applied += 1;
      return true;
    },

    headPose: () => pose,
    frame: () => latest,
    tally: () => ({ ...tally }),
  };
}
