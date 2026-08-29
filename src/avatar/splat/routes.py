"""Which route builds this person's splat, and why.

Both routes end in the same artefact - a Gaussian splat of one person - and the
customer chooses neither. Whatever they were able to find decides it:

    a usable video    RECONSTRUCT. One session shows the same face from many
                      angles as it moves, so every Gaussian comes from a camera
                      that actually saw them. The best likeness obtainable.
    photographs only  GENERATE. The unseen angles are invented from a single
                      anchor image by an image-to-3D model, with the remaining
                      photographs correcting skin and detail.
    neither           refuse, naming what is missing.

Photographs taken years apart cannot be *reconstructed* into one splat, and it
is not a volume problem: static splatting assumes one geometrically consistent
subject, and a person at forty and at fifty-five has no single 3D truth. Thirty
photographs across three decades will not beat one thirty-second video. That is
why the route is decided from what exists rather than offered as a setting. A
family asked to choose would choose wrong, in the direction that costs them the
likeness, and they would never know.

The decision is a value rather than a branch. It is recorded on the job, it
travels into the result, and it carries the sentence a support agent and the
customer are shown. A route chosen inside a function and then forgotten is a
route nobody can explain six months later when a family says it does not look
like him.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Shortest clip that can be reconstructed. Below this a head has not turned
# through enough angles for the frames to be different observations: it is one
# viewpoint held for a moment, which is a photograph that took longer to
# upload. Eight seconds at the half-second frame spacing in ingest/video.py
# yields about sixteen distinct views, just past the fifteen-image floor the
# photo checks already use for a stable identity.
MIN_VIDEO_SECONDS = 8.0

# How many of those frames must actually contain a face. A clip that spends
# most of itself on a ceiling or a shoulder is not multi-view coverage of a
# person, however long it runs.
MIN_VIDEO_VIEWS = 15

# The floor for generation. One anchor image is all TRELLIS or LGM needs, but a
# single photograph leaves nothing to correct it against: whatever the model
# invents about the skin stands unchallenged. Two more photographs is the least
# that lets the generated surface be pulled back towards the real person.
MIN_PHOTOS_FOR_GENERATION = 3


class Route(str, Enum):
    """How a splat gets built. Values are stable: they are recorded on a row."""

    RECONSTRUCT = "reconstruct"
    GENERATE = "generate"
    REFUSE = "refuse"


@dataclass(frozen=True)
class Intake:
    """What the customer actually uploaded, as the router needs to see it.

    Storage keys, never bytes. A thirty-image set weighs tens of megabytes and
    has no business passing through the process that is only deciding what to
    do with it.

    A video arrives with its frame counts because a clip nobody has looked at
    cannot be judged. Trusting a duration alone would let a thirty-second video
    of a shoulder be reconstructed, and the failure would only be visible in
    the finished avatar.
    """

    tenant_id: str
    photo_set_id: str
    # Accepted photographs only. Anything the ingest checks rejected is not
    # evidence of a face and must not be counted as one here.
    photo_keys: tuple[str, ...] = ()
    video_key: str | None = None
    video_seconds: float = 0.0
    video_frames: int = 0
    video_frames_with_face: int = 0
    # Which photograph anchors a generated build. Defaulted by the planner to
    # the first accepted image; supplied explicitly when a human has picked the
    # one that most looks like them, which is a judgement no metric makes well.
    anchor_key: str | None = None
    # Short edge of the source material, in pixels. Reported rather than
    # gated - the ingest checks already refused anything under 512 - because
    # the quality report should say what the likeness was built from.
    source_short_edge_px: int = 0

    def __post_init__(self) -> None:
        if self.video_key and self.video_frames <= 0:
            raise ValueError(
                "a video must arrive with the number of frames examined; a clip "
                "nobody has looked at cannot be judged usable"
            )
        if self.video_frames_with_face > self.video_frames:
            raise ValueError("more frames contained a face than were examined")

    @property
    def video_problem(self) -> str | None:
        """Why the video cannot be reconstructed, in plain words, or None.

        Returned as a sentence rather than a boolean because it is quoted
        verbatim to the customer when the build falls back to photographs. A
        family whose video was too short should be told that, not told nothing.
        """
        if not self.video_key:
            return "no video was uploaded"
        if self.video_seconds < MIN_VIDEO_SECONDS:
            return (
                f"the video is {self.video_seconds:.0f} seconds long, and we need at least "
                f"{MIN_VIDEO_SECONDS:.0f} seconds to see the face from more than one angle"
            )
        if self.video_frames_with_face < MIN_VIDEO_VIEWS:
            return (
                f"we could only find a face in {self.video_frames_with_face} frames of the "
                f"video, and we need at least {MIN_VIDEO_VIEWS}"
            )
        return None

    @property
    def views(self) -> int:
        """Source images the build optimises over."""
        return self.video_frames if self.video_problem is None else len(self.photo_keys)

    @property
    def views_with_face(self) -> int:
        """How many of those views a face was actually found in.

        For photographs the two numbers are equal by construction: an image
        with no face was never accepted. For a clip they differ, and the gap is
        worth reporting - it is the blinks, the turns away and the frames of a
        hand in front of the lens.
        """
        return (
            self.video_frames_with_face
            if self.video_problem is None
            else len(self.photo_keys)
        )


@dataclass(frozen=True)
class RouteDecision:
    """The route, the sentence that explains it, and the evidence behind it.

    Three audiences, three fields. `route` is for the pipeline, `reasoning` is
    for the customer and reads as English, and `considered` is the factual
    trail support needs when a family disputes the result. Keeping them in one
    frozen value is what stops the explanation drifting from the decision.
    """

    route: Route
    reasoning: str
    # Only on a refusal: what the customer must supply, itemised, in their
    # words rather than ours.
    missing: tuple[str, ...] = ()
    considered: tuple[str, ...] = ()

    @property
    def buildable(self) -> bool:
        return self.route is not Route.REFUSE


def _evidence(intake: Intake) -> tuple[str, ...]:
    """What was weighed, as facts. Support reads this, not the code."""
    trail = [
        f"video: {intake.video_seconds:.0f}s"
        if intake.video_key
        else "video: none uploaded",
        f"frames examined: {intake.video_frames}, with a face: {intake.video_frames_with_face}",
        f"photographs accepted: {len(intake.photo_keys)}",
    ]
    if intake.source_short_edge_px:
        trail.append(f"source short edge: {intake.source_short_edge_px}px")
    return tuple(trail)


def choose_route(intake: Intake) -> RouteDecision:
    """Decide how this person's splat gets built.

    The video wins whenever it is usable, even against a large album. Measured
    geometry beats invented geometry, and no number of photographs converts one
    into the other.
    """
    evidence = _evidence(intake)
    problem = intake.video_problem

    if problem is None:
        return RouteDecision(
            route=Route.RECONSTRUCT,
            reasoning=(
                "We used your video because it shows how they actually moved. One "
                "recording gives us the same face from many angles in a single "
                "session, so every part of the likeness comes from a camera that "
                "saw them rather than from a model's guess."
            ),
            considered=evidence,
        )

    photographs = len(intake.photo_keys)
    if photographs >= MIN_PHOTOS_FOR_GENERATION:
        others = photographs - 1
        return RouteDecision(
            route=Route.GENERATE,
            reasoning=(
                f"We built this from your photographs because {problem}. We anchored "
                "the likeness on one photograph and filled in the angles no camera "
                f"captured, with your other {others} photographs correcting the skin "
                "and detail. It will look right; the angles nobody photographed are "
                "not measured."
            ),
            considered=evidence,
        )

    # Neither route is open. Refusing is the product decision: a likeness built
    # from one blurry photograph is not a cheaper avatar, it is a stranger with
    # their haircut, and a family shown that once does not come back.
    missing = [
        (
            f"a video of at least {MIN_VIDEO_SECONDS:.0f} seconds where their face "
            f"is visible, because {problem}"
        ),
        (
            f"or at least {MIN_PHOTOS_FOR_GENERATION} photographs where their face "
            f"is clear and sharp, and we have {photographs}"
        ),
    ]
    return RouteDecision(
        route=Route.REFUSE,
        reasoning=(
            "We cannot build a likeness from what has been uploaded yet. We would "
            "rather tell you that than send you a face that is not theirs."
        ),
        missing=tuple(missing),
        considered=evidence,
    )
