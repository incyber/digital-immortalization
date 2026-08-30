"""The splat build: one job contract, two backends, one honest report.

A splat build is minutes of GPU on someone's dead father, which makes three
things non-negotiable and they are what this module exists to hold.

*Nothing goes through memory that does not have to.* Source assets travel as
storage keys inside one tenant's prefix. A thirty-image set is tens of
megabytes and the worker can fetch it itself; putting it in the job contract
would put it in every log line, every retry and every traceback.

*The GPU is the same GPU as everywhere else in this system.* RunPod Serverless,
following avatar/gpu/serverless.py exactly: nothing is allocated between jobs,
so there is no lifecycle to leak. The only thing added here is a bounded wait
and a cancel on the way out of a failure, and both are courtesies - the
platform's own execution timeout stops the worker whether or not this process
survives to ask.

*A route we have not shipped refuses in words.* One endpoint per route, because
the two routes are two different worker images. A route with no endpoint behind
it is answered the way insufficient material is - a sentence the customer can
act on, raised before anything is submitted - rather than attempted and failed.
Saying "we have not built that yet" is honest; a 500 is not, and a payload sent
to the wrong worker is worse than either.

*What was invented is stated, not discovered.* The photographs-only route fills
in angles no camera captured. That is the honest way to build a splat from an
album and it is also the thing a family must never learn later, so the measured
fraction is derived from the route rather than reported by the backend, and a
result cannot be constructed without a report at all.

The backend split is the one this project uses for the renderer and for
training: a real backend that submits to a GPU worker, and a fake that produces
a plausible result instantly. Everything around the GPU - routing, refusal,
cost, the report, the failure paths - is finished and tested with no GPU in the
room. When one appears, it is a swap rather than a rewrite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from loguru import logger

from avatar.gpu.serverless import (
    DEFAULT_EXECUTION_TIMEOUT_S,
    JobResult,
    JobState,
    ServerlessClient,
)
from avatar.splat.routes import (
    MIN_VIDEO_SECONDS,
    Intake,
    Route,
    RouteDecision,
    choose_route,
)
from avatar.storage.base import BlobStore
from avatar.storage.keys import asset_key, belongs_to

# One Gaussian in the compact .splat record is 32 bytes: three floats of
# position, three of scale, four bytes of colour and four of rotation. Size is
# tracked because it is the customer's download - the splat renders in their
# own browser, which is what removes the per-call GPU, and a file a phone will
# not fetch removes it by removing the product.
BYTES_PER_GAUSSIAN = 32

# Views before a head has camera support the whole way round. Twenty-four is
# one every fifteen degrees across the yaw range a talking head actually
# covers; past that a video is adding duplicates rather than angles.
VIEWS_FOR_FULL_COVERAGE = 24

# The most of a generated splat that may ever be called measured, however large
# the album. A photograph shows one hemisphere at best, so the back of the head
# is invented even when thirty pictures arrive. Capping this is the difference
# between an honest report and a flattering one.
MAX_MEASURED_ON_GENERATION = 0.6

# What a build without a GPU claims. Roughly what gsplat produces per view at
# these budgets, and roughly the rate it optimises at on a mid-range card -
# enough that a progress bar built against the fake behaves like the real one.
GAUSSIANS_PER_VIEW = 40_000
FAKE_ITERATIONS_PER_SECOND = 50.0


class SplatError(RuntimeError):
    """Anything that stops a splat existing."""


class SplatRefused(SplatError):
    """The material cannot produce a likeness, and we said so instead of trying.

    Carries the decision so a caller can show the customer the itemised list of
    what is missing rather than re-deriving it from a string.
    """

    def __init__(self, decision: RouteDecision):
        detail = " ".join(decision.missing)
        super().__init__(f"{decision.reasoning} We need {detail}" if detail else decision.reasoning)
        self.decision = decision


class SplatBuildError(SplatError):
    """The build was attempted and did not produce a splat."""


def route_unavailable(decision: RouteDecision) -> RouteDecision:
    """The refusal for a route we have not shipped, in our words not theirs.

    A route with no worker behind it is a limit of what we have built, and the
    customer must be told that rather than left to conclude their photographs
    were the problem. So this reads as our shortfall, and it still names the
    one thing they could do about it - because a refusal that offers no next
    step is a dead end even when it is honest.

    Returned as a REFUSE decision rather than raised as its own error on
    purpose. Every layer above already knows how to show a refusal: the
    service raises SplatRefused before a job row exists, and the HTTP surface
    answers 200 with guidance. A new exception type would need each of those
    taught again, and the family would meet a 500 in the meantime.

    The evidence trail from the original decision is carried through, so
    support still sees what was uploaded when a family asks why.
    """
    if decision.route is Route.GENERATE:
        return RouteDecision(
            route=Route.REFUSE,
            reasoning=(
                "We can only build a likeness from video at the moment. Building "
                "one from photographs alone is a route we have not finished yet, "
                "so this is a limit of what we have built rather than anything "
                "wrong with the photographs you sent."
            ),
            missing=(
                (
                    f"a video of at least {MIN_VIDEO_SECONDS:.0f} seconds where "
                    "their face is visible, which is the only material we can "
                    "build from today"
                ),
            ),
            considered=decision.considered,
        )

    # The reconstruct route, unconfigured. Nothing the customer can upload
    # changes it, so nothing is asked of them: `missing` stays empty and the
    # sentence they are shown is this one.
    return RouteDecision(
        route=Route.REFUSE,
        reasoning=(
            "We cannot build this likeness right now: the part of our system "
            "that reconstructs a face from video is not available. That is our "
            "problem rather than anything to do with what you uploaded, and "
            "nothing you sent has been lost. Please try again shortly."
        ),
        considered=decision.considered,
    )


class Quality(str, Enum):
    """How much GPU this build is worth. Values are recorded on a row.

    Three points rather than a slider because each is a different product
    decision, not a different number. PREVIEW is what a customer sees while
    deciding; STANDARD is what ships; HIGH exists for the one avatar a family
    will look at closely, and costs a download most phones will refuse.
    """

    PREVIEW = "preview"
    STANDARD = "standard"
    HIGH = "high"

    @property
    def gaussian_budget(self) -> int:
        return {
            Quality.PREVIEW: 200_000,    # ~6MB, fetches over a phone connection
            Quality.STANDARD: 800_000,   # ~25MB, the shipping default
            Quality.HIGH: 2_000_000,     # ~64MB, deliberately not the default
        }[self]

    @property
    def iterations(self) -> int:
        return {
            Quality.PREVIEW: 3_000,
            Quality.STANDARD: 15_000,
            Quality.HIGH: 30_000,
        }[self]

    @property
    def wait_s(self) -> float:
        """The longest this build waits before giving up and cancelling.

        Every value sits under the platform's own execution timeout on purpose.
        Waiting past that is waiting for a job the platform has already killed,
        which turns a clear failure into a hang.
        """
        return {
            Quality.PREVIEW: 240.0,
            Quality.STANDARD: 600.0,
            Quality.HIGH: float(DEFAULT_EXECUTION_TIMEOUT_S),
        }[self]


@dataclass(frozen=True)
class QualityReport:
    """What can be said about a finished splat with nobody looking at it.

    Every number here is either counted from the source material or derived
    from the route. Nothing is asked of the backend that the backend could
    flatter, which is why `measured_fraction` is a property and not a field: a
    worker cannot report a photographs-only build as fully measured, because
    there is nowhere to report it.
    """

    route: Route
    views_used: int
    views_with_face: int
    # Share of the viewing sphere the finished splat renders plausibly. High on
    # the generated route by construction - the model invents all the way
    # round - which is exactly why it must not be read as a quality score.
    # `measured_fraction` is the one that says whether any of it is true.
    angular_coverage: float
    source_short_edge_px: int = 0

    def __post_init__(self) -> None:
        if self.route is Route.REFUSE:
            raise ValueError("a refused build has no splat to report on")
        if self.views_used < 1:
            raise ValueError("a splat built from no views is not a splat")
        if not 0 <= self.views_with_face <= self.views_used:
            raise ValueError("views with a face must be between none and all of them")
        if not 0.0 <= self.angular_coverage <= 1.0:
            raise ValueError("angular coverage is a fraction of the viewing sphere")

    @property
    def measured_fraction(self) -> float:
        """How much of this likeness came from a camera that saw the person."""
        if self.route is Route.RECONSTRUCT:
            # Every Gaussian was optimised against a frame of the real person.
            # That is the whole reason this route wins whenever it is open.
            return 1.0
        return round(min(MAX_MEASURED_ON_GENERATION, self.views_used / VIEWS_FOR_FULL_COVERAGE), 2)

    @property
    def generated_fraction(self) -> float:
        return round(1.0 - self.measured_fraction, 2)

    @property
    def face_in_every_view(self) -> bool:
        return self.views_with_face == self.views_used

    @property
    def disclosure(self) -> str:
        """The sentence the customer is shown. Never empty, on either route.

        The plan requires that a partly invented likeness is stated up front
        rather than discovered later, so this is computed rather than passed
        in: there is no code path that can build a report without one.
        """
        if self.route is Route.RECONSTRUCT:
            return (
                "Every part of this likeness was reconstructed from your video. "
                "Nothing about the face was invented: each point of it comes from "
                "a frame where a camera saw them."
            )
        invented = round(self.generated_fraction * 100)
        return (
            f"About {invented}% of this likeness was generated rather than "
            f"photographed. Your {self.views_used} photographs fixed the face, skin "
            "and detail everywhere a camera looked; the angles no photograph covers "
            "were filled in by a model trained on many faces. It will look right. "
            "That part of it is not measured."
        )

    @property
    def concerns(self) -> tuple[str, ...]:
        """What is weak about this build, in the words support would use."""
        notes: list[str] = []
        if not self.face_in_every_view:
            missed = self.views_used - self.views_with_face
            notes.append(
                f"{missed} of the {self.views_used} source images had no face in "
                "them and contributed nothing to the likeness"
            )
        if self.source_short_edge_px and self.source_short_edge_px < 512:
            notes.append(
                f"the source material is {self.source_short_edge_px}px on its short "
                "edge, which carries less facial detail than the pipeline expects"
            )
        if self.angular_coverage < 0.5:
            notes.append(
                "the source only shows the face across a narrow range of angles, so "
                "the likeness is weakest seen from the side"
            )
        if self.route is Route.GENERATE and self.measured_fraction < 0.4:
            notes.append(
                "most of this likeness is generated; a video of the person would "
                "replace the invented part with measured geometry"
            )
        return tuple(notes)


@dataclass(frozen=True)
class SplatJob:
    """One splat build, resolved before a GPU is asked for anything.

    Source assets are storage keys, and that is enforced here rather than
    documented: bytes in this contract would mean a thirty-image set living in
    the orchestrator's memory for the length of a multi-minute build, and
    appearing in every log line that prints a job.

    Keys are also checked against the tenant prefix. A key arriving from a row
    that was somehow wrong is the one way a build could read another family's
    photographs, and this is the last place to stop it.
    """

    tenant_id: str
    avatar_id: str
    decision: RouteDecision
    quality: Quality
    output_key: str
    photo_keys: tuple[str, ...] = ()
    video_key: str | None = None
    anchor_key: str | None = None
    views: int = 1
    views_with_face: int = 1
    source_short_edge_px: int = 0

    def __post_init__(self) -> None:
        if not self.decision.buildable:
            raise SplatRefused(self.decision)
        for key in self.source_keys + (self.output_key,):
            if isinstance(key, bytes | bytearray | memoryview):
                raise TypeError(
                    "source assets travel as storage keys, never as bytes: a job "
                    "carrying image data puts every uploaded photograph in memory "
                    "and in the logs for the length of the build"
                )
            if not isinstance(key, str):
                raise TypeError(f"a storage key must be a string; got {type(key).__name__}")
            if not belongs_to(key, self.tenant_id):
                raise ValueError(f"{key!r} is outside tenant {self.tenant_id}'s prefix")
        if self.anchor_key and self.anchor_key not in self.photo_keys:
            raise ValueError("the anchor photograph must be one of the photographs supplied")
        if self.views < 1:
            raise ValueError("a build with no source views cannot produce a likeness")
        if not 0 <= self.views_with_face <= self.views:
            raise ValueError("views with a face must be between none and all of them")

    @property
    def route(self) -> Route:
        return self.decision.route

    @property
    def source_keys(self) -> tuple[str, ...]:
        keys = list(self.photo_keys)
        if self.video_key:
            keys.append(self.video_key)
        return tuple(keys)

    def payload(self) -> dict:
        """What the worker is sent. Keys and numbers, nothing heavier."""
        return {
            "task": "splat",
            "route": self.route.value,
            "tenant_id": self.tenant_id,
            "avatar_id": self.avatar_id,
            "video_key": self.video_key,
            "photo_keys": list(self.photo_keys),
            "anchor_key": self.anchor_key,
            "output_key": self.output_key,
            "gaussian_budget": self.quality.gaussian_budget,
            "iterations": self.quality.iterations,
        }


@dataclass(frozen=True)
class SplatResult:
    """A built splat and everything that must be said about it.

    `decision` and `report` have no defaults on purpose. The route that built a
    likeness and the share of it that was invented are not optional metadata to
    be filled in later; a result that omits either cannot be constructed.
    """

    decision: RouteDecision
    report: QualityReport
    splat_key: str
    gaussian_count: int
    size_bytes: int
    build_seconds: float
    cost_usd: float
    backend: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def route(self) -> Route:
        return self.decision.route

    @property
    def reasoning(self) -> str:
        return self.decision.reasoning


def cost_of(execution_ms: int) -> float:
    """Priced exactly as every other GPU job in this system is priced.

    Delegated rather than reimplemented so a change of card or of provider
    moves one number, and so a splat build cannot quietly develop its own
    cheaper-looking arithmetic.
    """
    return JobResult(id="", state=JobState.COMPLETED, execution_ms=execution_ms).cost


def estimated_coverage(route: Route, views: int) -> float:
    """Share of the viewing sphere the result renders, when nobody measured it.

    A generated splat covers every direction by construction - that is what
    generation does, and it is why coverage is not a quality score. A
    reconstruction covers only what the head turned through.
    """
    if route is Route.GENERATE:
        return 1.0
    return round(min(1.0, views / VIEWS_FOR_FULL_COVERAGE), 2)


def plan(intake: Intake, avatar_id: str, *, quality: Quality = Quality.STANDARD) -> SplatJob:
    """Decide the route and resolve the job, or refuse.

    Refusal happens here, before anything is submitted, so an unbuildable set
    never costs a GPU second.
    """
    decision = choose_route(intake)
    if not decision.buildable:
        raise SplatRefused(decision)

    anchor = None
    if decision.route is Route.GENERATE:
        # The anchor is the photograph the invented geometry is built around,
        # so it is the single most consequential choice on this route. A
        # caller's explicit pick wins; the fallback is the first accepted
        # image rather than a scored one, because no metric picks "the one
        # that looks most like him" better than the family already did.
        anchor = intake.anchor_key or intake.photo_keys[0]

    return SplatJob(
        tenant_id=intake.tenant_id,
        avatar_id=avatar_id,
        decision=decision,
        quality=quality,
        output_key=asset_key(intake.tenant_id, avatar_id, "avatar.splat"),
        photo_keys=tuple(intake.photo_keys),
        video_key=intake.video_key if decision.route is Route.RECONSTRUCT else None,
        anchor_key=anchor,
        views=intake.views,
        views_with_face=intake.views_with_face,
        source_short_edge_px=intake.source_short_edge_px,
    )


def report_for(job: SplatJob, angular_coverage: float | None = None) -> QualityReport:
    """The report both backends produce, built the same way from the same job.

    Shared rather than written twice: if each backend assembled its own report,
    the fake and the real one would eventually disagree about how much of a
    likeness was invented, and only one of them would be shown to a customer.
    """
    coverage = (
        estimated_coverage(job.route, job.views) if angular_coverage is None else angular_coverage
    )
    return QualityReport(
        route=job.route,
        views_used=job.views,
        views_with_face=job.views_with_face,
        angular_coverage=max(0.0, min(1.0, float(coverage))),
        source_short_edge_px=job.source_short_edge_px,
    )


@runtime_checkable
class SplatBackend(Protocol):
    """Where a splat is actually optimised.

    Split into submit and collect rather than one blocking call, and that split
    is the safety property: the caller holds the job id before it starts
    waiting, so a timeout, a crash or a cancelled task can name the exact job
    to stop. A single build() that returns only on success has a window where
    something is running and nothing knows its id.
    """

    @property
    def name(self) -> str:
        """Recorded on the job row, so a build can be traced to a backend."""

    def supports(self, route: Route) -> bool:
        """Whether this backend has somewhere to send this route's work.

        Asked before anything is submitted. A backend that cannot serve a
        route says so here and is refused in words, rather than discovering it
        at submit and surfacing as a failed build the customer cannot read.
        """

    async def submit(self, job: SplatJob) -> str:
        """Queue the build. Returns immediately with an external id."""

    async def collect(self, external_id: str, job: SplatJob, *, wait_s: float) -> SplatResult:
        """Wait for the build, bounded by wait_s. Raises SplatBuildError.

        Must cancel the job before raising on timeout: leaving a worker running
        while nothing is watching it is the failure this system is built to
        make impossible.
        """

    async def cancel(self, external_id: str) -> None:
        """Stop a build. Must be safe on one that has already finished."""


class FakeSplatBackend:
    """A splat build with no GPU, instantly, with a plausible result.

    Not a stub in the sense of being unfinished. It is the development backend,
    exactly as the viseme renderer and the local training runner are, and it
    implements the whole protocol so the flows around a build - refusal, the
    progress bar, cost display, the disclosure a customer reads, every failure
    path - are finished and tested before any GPU exists.

    It never sleeps. Simulated build seconds are reported as a number, because
    a test suite that waits five real minutes to prove a splat took five
    minutes is a test suite nobody runs.
    """

    def __init__(self, store: BlobStore | None = None, *, fail_in: str | None = None):
        # A store is optional. Given one, a placeholder artefact is written so
        # the download and attach paths can be exercised end to end.
        self._store = store
        # "submit" or "collect": which half fails. The failure paths around a
        # GPU are the ones that most need testing and the least likely to be
        # reachable on demand from a real provider.
        self._fail_in = fail_in
        self._counter = 0
        self.submitted: list[SplatJob] = []
        self.cancelled: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    def supports(self, route: Route) -> bool:
        """Both routes, always. There is no image to be missing.

        Which is exactly why the development backend is the one the whole
        customer flow is built against: an unshipped worker must not make a
        route untestable.
        """
        return True

    async def submit(self, job: SplatJob) -> str:
        if self._fail_in == "submit":
            raise SplatBuildError("the fake backend was asked to fail at submit")
        self._counter += 1
        external_id = f"fake-{self._counter:06d}"
        self.submitted.append(job)
        return external_id

    async def collect(self, external_id: str, job: SplatJob, *, wait_s: float) -> SplatResult:
        if self._fail_in == "collect":
            raise SplatBuildError("the fake backend was asked to fail at collect")

        gaussians = min(job.quality.gaussian_budget, job.views * GAUSSIANS_PER_VIEW)
        size_bytes = gaussians * BYTES_PER_GAUSSIAN
        build_seconds = job.quality.iterations / FAKE_ITERATIONS_PER_SECOND

        if self._store is not None:
            # A marker, not 25MB of zeros. The size reported is what a real
            # build of this many Gaussians would weigh; writing that many bytes
            # into a temp directory is a slow way to prove nothing.
            await self._store.put(
                job.tenant_id,
                job.output_key,
                f"placeholder splat: {gaussians} gaussians, {job.route.value}".encode(),
                "application/octet-stream",
            )

        report = report_for(job)
        return SplatResult(
            decision=job.decision,
            report=report,
            splat_key=job.output_key,
            gaussian_count=gaussians,
            size_bytes=size_bytes,
            build_seconds=build_seconds,
            cost_usd=cost_of(int(build_seconds * 1000)),
            backend=self.name,
            warnings=report.concerns,
        )

    async def cancel(self, external_id: str) -> None:
        self.cancelled.append(external_id)


class RunPodSplatBackend:
    """The real build, on RunPod Serverless. One endpoint per route.

    No second GPU approach: this is avatar/gpu/serverless.py, with the same
    guarantees doing the same work. Nothing is allocated between jobs, the
    platform's execution timeout stops a hung worker without this process being
    alive, and the cancel on the way out of a failure is a courtesy on top of
    that rather than the thing being relied on.

    Two endpoints rather than one because the two routes are two different
    workers. Reconstruction fits Gaussians to the frames of a video; generation
    runs an image-to-3D model over a single anchor photograph. Different
    weights, different card, different runtime, different image - a single
    endpoint id could only ever have been right for one of them, and pointing
    the other at it would submit a payload the worker does not understand.

    An endpoint that is not configured is a first-class state, not an error:
    `supports` is asked before anything is submitted, so a route we have not
    shipped is refused in a sentence rather than discovered as a failed build.

    The client is injectable so the whole of this class - the poll loop, the
    timeout, the cancel, the parsing - is tested without a network or an
    account. An injected client stands in for every endpoint, because none of
    those behaviours differ by route.
    """

    def __init__(
        self,
        api_key: str = "",
        reconstruct_endpoint_id: str = "",
        generate_endpoint_id: str = "",
        *,
        client: ServerlessClient | None = None,
        poll_s: float = 2.0,
    ):
        self._api_key = api_key
        self._endpoints: dict[Route, str] = {
            Route.RECONSTRUCT: reconstruct_endpoint_id,
            Route.GENERATE: generate_endpoint_id,
        }
        self._override = client
        # Built on first use rather than in the constructor, so a deployment
        # that has one of the two endpoints starts and serves the route it can
        # instead of refusing to start at all.
        self._clients: dict[Route, ServerlessClient] = {}
        # Which route each submitted job went out on, so a cancel reaches the
        # endpoint that is running it. A cancel sent to the other endpoint is a
        # cancel that does not happen, and the whole posture here is that a
        # failure leaves nothing running.
        self._routes: dict[str, Route] = {}
        self._poll_s = poll_s

    @property
    def name(self) -> str:
        return "runpod"

    def supports(self, route: Route) -> bool:
        return self._override is not None or bool(self._endpoints.get(route, ""))

    def endpoint_for(self, route: Route) -> str:
        """Which endpoint this route's work goes to. Empty when none is set."""
        return self._endpoints.get(route, "")

    def _client_for(self, route: Route) -> ServerlessClient:
        if self._override is not None:
            return self._override
        endpoint = self._endpoints.get(route, "")
        if not endpoint:
            # Only reachable if something submitted without asking supports()
            # first. Raised rather than refused because by this point it is a
            # programming error, not a thing to tell a customer.
            raise SplatBuildError(
                f"no serverless endpoint is configured for the {route.value} route"
            )
        client = self._clients.get(route)
        if client is None:
            client = ServerlessClient(self._api_key, endpoint)
            self._clients[route] = client
        return client

    async def submit(self, job: SplatJob) -> str:
        # On a thread because the client is blocking and the caller is usually
        # the gateway's event loop. The first version of the sibling GPU path
        # polled inline and stopped every other request in the process,
        # including the one asking how the build was going.
        client = self._client_for(job.route)
        external_id = await asyncio.to_thread(client.submit, job.payload())
        self._routes[external_id] = job.route
        return external_id

    async def collect(self, external_id: str, job: SplatJob, *, wait_s: float) -> SplatResult:
        client = self._client_for(job.route)
        result = await asyncio.to_thread(self._wait, client, external_id, wait_s)
        # Only on the way out with a result: a build that raised is about to be
        # cancelled by the builder, and the cancel needs this to find it.
        self._routes.pop(external_id, None)
        return self._result_from(result, job)

    async def cancel(self, external_id: str) -> None:
        route = self._routes.pop(external_id, None)
        if route is None:
            # A job this process did not submit - a gateway restart, most
            # likely. There is no way to know which endpoint holds it, and
            # guessing would send the cancel to the wrong one. The platform's
            # own execution timeout stops it, which is the guarantee that was
            # doing the real work anyway.
            logger.warning(
                f"cannot cancel splat build {external_id}: this process does not "
                "know which endpoint it was submitted to"
            )
            return
        await asyncio.to_thread(self._client_for(route).cancel, external_id)

    def _wait(self, client: ServerlessClient, external_id: str, wait_s: float) -> JobResult:
        """Poll until terminal, then stop caring; cancel and raise on timeout.

        Written here rather than using ServerlessClient.run because run submits
        and waits in one call. This needs the id first, so that a build which
        outlives its bound can be cancelled by name.
        """
        deadline = time.monotonic() + wait_s
        while True:
            result = client.status(external_id)
            if result.state.terminal:
                return result
            if time.monotonic() >= deadline:
                client.cancel(external_id)
                raise SplatBuildError(
                    f"splat build {external_id} did not finish within {wait_s:.0f}s"
                )
            time.sleep(self._poll_s)

    def _result_from(self, result: JobResult, job: SplatJob) -> SplatResult:
        if result.state is not JobState.COMPLETED:
            raise SplatBuildError(
                f"splat build {result.id} {result.state.value}: {result.error or 'no reason given'}"
            )

        output = result.output or {}
        if output.get("error"):
            raise SplatBuildError(f"splat build failed: {output['error']}")

        splat_key = output.get("splat_key")
        if not splat_key:
            # A completed job with no artefact is a failure that would
            # otherwise be recorded as a success and discovered on the call.
            raise SplatBuildError(f"splat build {result.id} finished with no splat")
        if splat_key != job.output_key:
            raise SplatBuildError(
                f"the worker wrote {splat_key!r} rather than {job.output_key!r}"
            )

        gaussians = int(output.get("gaussians") or 0)
        report = report_for(job, output.get("angular_coverage"))
        logger.info(
            f"splat built via {job.route.value}: {gaussians} gaussians in "
            f"{result.execution_ms}ms (~${result.cost:.4f}), "
            f"{report.measured_fraction:.0%} measured"
        )
        return SplatResult(
            decision=job.decision,
            report=report,
            splat_key=splat_key,
            gaussian_count=gaussians,
            size_bytes=int(output.get("bytes") or gaussians * BYTES_PER_GAUSSIAN),
            build_seconds=result.execution_ms / 1000.0,
            cost_usd=result.cost,
            backend=self.name,
            warnings=report.concerns,
        )


class SplatBuilder:
    """Routes, refuses, submits, waits, and leaves nothing running.

    The invariant worth stating: after any call to build(), successful or not,
    `pending` is empty. A job id is held only for the window in which something
    is waiting on it, and every exit from that window - failure, timeout,
    cancelled task, Ctrl-C - passes through the same cancel.
    """

    def __init__(self, backend: SplatBackend):
        self._backend = backend
        self._pending: dict[str, SplatJob] = {}

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def pending(self) -> tuple[str, ...]:
        """Jobs this builder is currently waiting on. Empty between builds."""
        return tuple(self._pending)

    def supports(self, route: Route) -> bool:
        """Whether the backend behind this builder can build that route.

        Exposed so a caller can refuse before it writes a job row, rather than
        starting a build it will have to fail a moment later.
        """
        return self._backend.supports(route)

    async def build(
        self,
        intake: Intake,
        avatar_id: str,
        *,
        quality: Quality = Quality.STANDARD,
    ) -> SplatResult:
        """Build this person's splat, or say why it cannot be built."""
        job = plan(intake, avatar_id, quality=quality)
        if not self._backend.supports(job.route):
            # Refused before submit, so nothing is allocated, nothing is
            # pending and there is nothing to cancel. The material was fine;
            # what is missing is a worker of ours, and route_unavailable says
            # so in those words.
            raise SplatRefused(route_unavailable(job.decision))
        logger.info(
            f"splat for avatar {avatar_id}: {job.route.value} at {quality.value} "
            f"from {job.views} views"
        )

        external_id = await self._backend.submit(job)
        self._pending[external_id] = job
        try:
            result = await self._backend.collect(external_id, job, wait_s=quality.wait_s)
        except BaseException:
            # BaseException rather than Exception: a cancelled task and a
            # Ctrl-C are exactly the cases where a GPU would otherwise be left
            # running with nobody watching it.
            await self._stop(external_id)
            raise
        finally:
            self._pending.pop(external_id, None)

        logger.info(
            f"splat {result.splat_key}: {result.gaussian_count} gaussians, "
            f"{result.size_bytes // 1024}KB, {result.build_seconds:.0f}s, "
            f"~${result.cost_usd:.4f}"
        )
        return result

    async def _stop(self, external_id: str) -> None:
        """Cancel, and never let the cancel hide the failure that caused it."""
        try:
            await self._backend.cancel(external_id)
        except Exception as exc:  # noqa: BLE001 - a failed cancel must not mask the real error
            logger.warning(f"cancel of splat build {external_id} did not take: {exc}")
