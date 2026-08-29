"""What the likeness is doing, at any instant, and why.

This is where the layers meet. Five of them are summed for every frame:

    base      the resting posture of this particular person
    affect    who they are right now - warm, grave, amused
    prosody   how they are saying this sentence - where the stress falls
    gesture   discrete things they are doing - a nod, a shrug, a lean
    drift     the aperiodic motion of any living body

Two decisions shape everything else.

**Time is measured in speech, not in seconds elapsed.** The model writes a
sentence before the voice synthesises it, and the voice synthesises it before
the renderer draws it. So a nod that must peak on a stressed syllable can be
started before that syllable is audible. Timed on a wall clock, every nod
lands after the beat it was meant to hit.

**pose_at is pure.** No waiting, no allocation beyond one frame, no state to
mutate except the spring integrator. That is what lets barge-in be instant -
there is no motion thread to join and no queue to drain, so interrupting is a
state write measured in microseconds rather than a cancellation that has to
propagate. It is also why the whole system is testable on a laptop: the output
is numbers, and numbers can be asserted on.

Backchannel is the one thing here that runs on the wall clock instead, and
deliberately. A nod acknowledging what somebody just said belongs 150-250ms
after they said it; anchored tightly, it would be uncanny rather than attentive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from avatar.motion.affect import Affect, affect_from_label, blend
from avatar.motion.noise import BlinkScheduler, PinkDrift, SaccadeScheduler
from avatar.motion.pose import BY_NAME, CHANNELS, PoseFrame

# How hard each channel is pulled towards its target. Critically damped, so
# nothing ever overshoots and nothing ever cuts. The face is stiffer than the
# body because a slow brow reads as sedated and a fast torso reads as twitchy.
STIFFNESS = {
    "head_yaw": 12.0, "head_pitch": 12.0, "head_roll": 10.0,
    "blink": 60.0,
    "lid_upper_l": 18.0, "lid_upper_r": 18.0,
    "brow_inner_l": 20.0, "brow_inner_r": 20.0,
    "brow_outer_l": 20.0, "brow_outer_r": 20.0,
    "jaw_open": 40.0,
    "mouth_smile_l": 8.0, "mouth_smile_r": 8.0, "mouth_press": 10.0,
    "torso_lean": 6.0, "torso_yaw": 6.0, "shoulder_raise": 8.0,
    "breath": 6.0,
}

# Gaze is exempt from smoothing entirely. Real eyes are ballistic - still, then
# somewhere else, with nothing visible in between - and interpolating between
# fixations is the most reliably uncanny thing an animated face can do.
UNSMOOTHED = {"gaze_yaw", "gaze_pitch"}

# Breathing never stops, in any state. It is the cheapest signal that something
# is alive, and its absence is what makes a paused avatar read as a photograph.
BREATH_HZ = {"listening": 0.22, "thinking": 0.20, "speaking": 0.30, "waiting": 0.18}

# How much of the head's range the drift is allowed. Measured against real
# motion templates elsewhere in this project rather than chosen by taste.
DRIFT_SCALE = {"head_yaw": 0.055, "head_pitch": 0.035, "head_roll": 0.018}


class Attitude(StrEnum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    WAITING = "waiting"


# What each state does with the face, beyond whatever it is saying. These are
# the differences that make listening read as listening rather than as waiting
# for input: attention is mostly brow and lean, not motion.
POSTURE = {
    Attitude.LISTENING: {"brow_inner_l": 0.10, "brow_inner_r": 0.10,
                         "lid_upper_l": 0.08, "lid_upper_r": 0.08,
                         "torso_lean": 0.15},
    Attitude.THINKING: {"brow_inner_l": 0.20, "brow_inner_r": 0.20,
                        "gaze_yaw": -0.18, "gaze_pitch": 0.12},
    Attitude.SPEAKING: {},
    Attitude.WAITING: {"lid_upper_l": -0.05, "lid_upper_r": -0.05,
                       "torso_lean": -0.05},
}

BLINK_RATE = {"listening": 18.0, "thinking": 22.0, "speaking": 11.0, "waiting": 15.0}


@dataclass
class _Spring:
    """One channel, moving towards its target without overshoot or cuts."""

    value: float = 0.0
    velocity: float = 0.0

    def step(self, target: float, omega: float, dt: float) -> float:
        # Critically damped: damping is exactly 2*omega, so the value
        # approaches the target and stops rather than ringing around it.
        accel = omega * omega * (target - self.value) - 2.0 * omega * self.velocity
        self.velocity += accel * dt
        self.value += self.velocity * dt
        return self.value


@dataclass
class MotionDirector:
    """A MotionSource: what the face and body are doing at time t."""

    session_seed: int = 0
    fps: int = 25

    # Neutral until the model says otherwise. source="carry" rather than
    # "tag", because nothing has been read yet and a later log should not
    # suggest the model asked for this.
    _affect: Affect = field(
        default_factory=lambda: affect_from_label(
            "neutral", 1.0, confidence=1.0, source="carry"
        )
    )
    _attitude: Attitude = Attitude.LISTENING
    _t: float = 0.0
    _generation: int = 0

    def __post_init__(self) -> None:
        self._drift = {
            name: PinkDrift(seed=self.session_seed + i, step_s=1.0 / self.fps)
            for i, name in enumerate(DRIFT_SCALE)
        }
        self._blinks = BlinkScheduler(seed=self.session_seed)
        self._saccades = SaccadeScheduler(seed=self.session_seed)
        self._springs = {c.name: _Spring() for c in CHANNELS}
        self._blink_until = -1.0
        self._gaze = (0.0, 0.0)
        self._events: list = []

    # ------------------------------------------------------------------
    @property
    def timeline(self) -> float:
        return self._t

    def set_attitude(self, attitude: Attitude, t: float) -> None:
        self._attitude = attitude
        self._t = max(self._t, t)

    def set_affect(self, affect: Affect) -> None:
        self._affect = affect

    def interrupt(self, t: float) -> None:
        """Barge-in. A state write, not a cancellation.

        Everything planned after t is dropped and the attitude returns to
        listening. Nothing is awaited, nothing is joined - which is why the
        renderer's 100ms cancel bound is never in danger from this side.
        """
        self._generation += 1
        self._events = [e for e in self._events if getattr(e, "t", 0.0) <= t]
        self._attitude = Attitude.LISTENING
        self._t = max(self._t, t)

    # ------------------------------------------------------------------
    def pose_at(self, t: float) -> PoseFrame:
        """The pose at t. Pure, deterministic, and never blocking."""
        dt = 1.0 / self.fps
        state = self._attitude.value

        targets = {c.name: 0.0 for c in CHANNELS}
        targets.update(POSTURE.get(self._attitude, {}))

        # Affect, as a resting offset rather than an expression played on top:
        # a sad person is not neutral plus sadness, they sit differently.
        a = self._affect
        targets["head_pitch"] += -0.06 * max(0.0, -a.valence)
        targets["brow_inner_l"] += 0.35 * max(0.0, -a.valence)
        targets["brow_inner_r"] += 0.35 * max(0.0, -a.valence)
        targets["brow_outer_l"] += 0.30 * a.arousal
        targets["brow_outer_r"] += 0.30 * a.arousal
        targets["mouth_smile_l"] += 0.45 * max(0.0, a.valence)
        targets["mouth_smile_r"] += 0.45 * max(0.0, a.valence)
        targets["lid_upper_l"] += 0.25 * (a.arousal - 0.35)
        targets["lid_upper_r"] += 0.25 * (a.arousal - 0.35)
        targets["torso_lean"] += 0.20 * (a.dominance - 0.5)

        for name, scale in DRIFT_SCALE.items():
            targets[name] += self._drift[name].advance_to(t) * scale

        targets["breath"] = 0.5 + 0.5 * math.sin(2 * math.pi * BREATH_HZ[state] * t)

        pose = PoseFrame(t=t)
        for channel in CHANNELS:
            name = channel.name
            if name in UNSMOOTHED:
                continue
            omega = STIFFNESS.get(name, 10.0)
            setattr(pose, name, self._springs[name].step(targets[name], omega, dt))

        # Blink is an event, not a curve: a schedule decides when, and the
        # shape is a fixed fast close and open. Smoothing it towards a target
        # produces a droop rather than a blink.
        for blink in self._blinks.blinks_until(t, BLINK_RATE[state]):
            self._blink_until = blink.t + blink.duration
        pose.blink = 1.0 if t < self._blink_until else 0.0

        # Gaze jumps and holds. Not smoothed, by the rule above.
        for saccade in self._saccades.saccades_until(t):
            self._gaze = (saccade.yaw, saccade.pitch)
            forced = self._blinks.force(saccade.t)
            if forced is not None:
                self._blink_until = forced.t + forced.duration
        pose.gaze_yaw = self._gaze[0] + targets["gaze_yaw"]
        pose.gaze_pitch = self._gaze[1] + targets["gaze_pitch"]

        self._t = max(self._t, t)
        return pose.clamped()

    # ------------------------------------------------------------------
    def blend_affect_towards(self, target: Affect, dt: float, **kwargs) -> Affect:
        """Move the affect, rate limited. See affect.blend for why slowly."""
        self._affect = blend(self._affect, target, dt, **kwargs)
        return self._affect

    def trace(self, seconds: float) -> np.ndarray:
        """Every channel over a span, for tests and for reviewing a change.

        Motion is reviewed as curves rather than pixels. It is the only way a
        person can judge a change to this system without a GPU, and a diff of
        numbers is a far better record of what changed than a video is.
        """
        frames = int(seconds * self.fps)
        return np.array([self.pose_at(i / self.fps).to_array() for i in range(frames)])


def slew_of(name: str) -> float:
    return BY_NAME[name].slew
