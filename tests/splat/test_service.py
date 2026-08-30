"""What a family gets when they upload what they have.

The route selector and the builder are tested next door. This suite defends
the part between them and the customer: that what was actually uploaded is
counted honestly, that a build is a job somebody can be told about afterwards,
that a failure leaves evidence rather than a bar that never moves, and above
all that the sentence saying how much of their father was invented is written
down where they will read it.

Nothing here needs a GPU. That is the point of the development backend: every
one of these behaviours is finished and defended before a card is rented.
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from avatar.config import Settings
from avatar.gateway.csrf import REQUIRED_HEADER, REQUIRED_VALUE
from avatar.gateway.models import (
    Avatar,
    Base,
    Photo,
    PhotoSet,
    PhotoSetStatus,
    TrainingJob,
    TrainingStatus,
    User,
)
from avatar.gateway.routes_splat import build_router
from avatar.gateway.tenancy import TenantError
from avatar.gpu.serverless import JobResult, JobState
from avatar.splat.build import (
    FakeSplatBackend,
    Quality,
    RunPodSplatBackend,
    SplatBuilder,
    SplatRefused,
    route_unavailable,
)
from avatar.splat.routes import Intake, Route, choose_route
from avatar.splat.service import (
    PROVIDER_PREFIX,
    SplatService,
    SplatUnavailable,
    build_splat_builder,
    intake_for,
)
from avatar.storage.keys import photo_key, source_clip_key
from avatar.storage.local import LocalBlobStore


def a_clip(seconds: float, fps: int = 6, size: tuple[int, int] = (320, 240)) -> bytes:
    """A real, decodable video of the requested length.

    Real rather than a handful of bytes with an .mp4 name, because the length
    is read back with ffprobe and the whole question this fixture exists to
    ask is whether an eight-second clip is treated differently from a
    four-second one.
    """
    width, height = size
    target = Path(tempfile.mkdtemp()) / "clip.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rng = np.random.default_rng(7)
    for _ in range(int(seconds * fps)):
        writer.write(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    writer.release()
    return target.read_bytes()


@pytest_asyncio.fixture
async def engine(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/splat.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def sessions(engine):
    """A source of sessions, shaped exactly like the gateway's get_db.

    The service is handed this rather than a session because a build finishes
    long after the request that started it has closed its own.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def source():
        async with factory() as session:
            yield session

    return source


@pytest_asyncio.fixture
async def db(sessions):
    source = sessions()
    session = await source.__anext__()
    yield session
    await source.aclose()


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(str(tmp_path / "blobs"))


@pytest_asyncio.fixture
async def owner(db):
    user = User(email="family@example.com", password_hash="x")
    db.add(user)
    await db.commit()
    return user


@pytest_asyncio.fixture
async def other_owner(db):
    """A second tenant. Every isolation test needs somebody to be excluded."""
    user = User(email="stranger@example.com", password_hash="x")
    db.add(user)
    await db.commit()
    return user


@pytest_asyncio.fixture
async def photo_set(db, owner):
    avatar = Avatar(owner_id=owner.id, display_name="Aurelio Sandoval", country="ES")
    db.add(avatar)
    await db.flush()
    a_set = PhotoSet(owner_id=owner.id, avatar_id=avatar.id, status=PhotoSetStatus.READY)
    db.add(a_set)
    await db.commit()
    return a_set


async def add_photos(db, photo_set, count: int, *, accepted: bool = True, frames: bool = False):
    """Rows as the ingest checks would have left them.

    Written directly because what is under test is what the counts decide, not
    face detection, which has its own suite.
    """
    for index in range(count):
        name = f"frame-{index:04d}.jpg" if frames else f"photo-{index:03d}.jpg"
        db.add(
            Photo(
                photo_set_id=photo_set.id,
                owner_id=photo_set.owner_id,
                blob_key=photo_key(photo_set.owner_id, photo_set.id, name),
                filename=name,
                content_type="image/jpeg",
                size_bytes=1,
                accepted=accepted,
                face_height_fraction=0.4,
            )
        )
    await db.commit()


async def add_clip(store, photo_set, seconds: float):
    await store.put(
        photo_set.owner_id,
        source_clip_key(photo_set.owner_id, photo_set.id),
        a_clip(seconds),
        "video/mp4",
    )


def service_for(store, sessions, **kwargs) -> SplatService:
    backend = kwargs.pop("backend", None) or FakeSplatBackend(store)
    return SplatService(SplatBuilder(backend), store, sessions, **kwargs)


async def finished(service, db, job_id, owner_id, timeout_s: float = 5.0) -> dict:
    await service.wait(job_id, timeout_s)
    return await service.read(db, job_id, owner_id)


# --- what the customer uploaded, counted honestly -------------------------


async def test_a_usable_clip_is_reconstructed_from_rather_than_generated(
    db, store, sessions, photo_set
):
    await add_clip(store, photo_set, seconds=12.0)
    await add_photos(db, photo_set, 20, frames=True)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)

    assert started.decision.route is Route.RECONSTRUCT
    result = await finished(service, db, started.job_id, photo_set.owner_id)
    assert result["status"] == "succeeded"
    assert result["route"] == "reconstruct"
    assert result["measured_fraction"] == 1.0


async def test_photographs_alone_are_generated_from(db, store, sessions, photo_set):
    await add_photos(db, photo_set, 20)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)

    assert started.decision.route is Route.GENERATE
    result = await finished(service, db, started.job_id, photo_set.owner_id)
    assert result["route"] == "generate"


async def test_a_clip_too_short_to_reconstruct_falls_back_to_the_frames_it_yielded(
    db, store, sessions, photo_set
):
    """Four seconds is one viewpoint held for a moment, not multi-view coverage."""
    await add_clip(store, photo_set, seconds=4.0)
    await add_photos(db, photo_set, 8, frames=True)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)

    assert started.decision.route is Route.GENERATE
    # And the customer is told which shortcoming sent them down this route,
    # in the measurement that caused it.
    assert "4 seconds" in started.decision.reasoning


async def test_a_clip_is_never_judged_usable_without_the_frames_that_prove_someone_looked(
    db, store, sessions, photo_set
):
    """Intake demands frame counts alongside a video, and this honours it.

    A stored clip that yielded no frames is a clip nobody has examined. It is
    reported as no usable video rather than defaulted to zero frames and
    smuggled past a check that exists precisely to stop that.
    """
    await add_clip(store, photo_set, seconds=30.0)
    await add_photos(db, photo_set, 20)

    intake = await intake_for(db, store, photo_set, photo_set.owner_id)

    assert intake.video_key is None
    assert intake.video_frames == 0

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    assert started.decision.route is Route.GENERATE


async def test_only_images_the_checks_accepted_are_counted(db, store, sessions, photo_set):
    await add_photos(db, photo_set, 2)
    await add_photos(db, photo_set, 9, accepted=False)

    intake = await intake_for(db, store, photo_set, photo_set.owner_id)

    assert len(intake.photo_keys) == 2


# --- refusal is guidance, not an error ------------------------------------


async def test_too_little_material_refuses_with_a_sentence_naming_what_is_missing(
    db, store, sessions, photo_set
):
    await add_photos(db, photo_set, 2)
    service = service_for(store, sessions)

    with pytest.raises(SplatRefused) as refusal:
        await service.start(db, photo_set.id, photo_set.owner_id)

    missing = " ".join(refusal.value.decision.missing)
    assert "at least 8 seconds" in missing
    assert "at least 3 photographs" in missing
    # In the customer's own counts, so the sentence tells them how far off
    # they are rather than only what the rule is.
    assert "we have 2" in missing


async def test_a_refusal_costs_nothing_and_leaves_no_job_to_explain(
    db, store, sessions, photo_set
):
    await add_photos(db, photo_set, 1)
    service = service_for(store, sessions)

    with pytest.raises(SplatRefused):
        await service.start(db, photo_set.id, photo_set.owner_id)

    assert service.running == ()
    jobs = (await db.execute(TrainingJob.__table__.select())).all()
    assert jobs == []


# --- the truth about the result reaches the customer ----------------------


async def test_a_generated_likeness_states_how_much_of_it_was_invented(
    db, store, sessions, photo_set
):
    """The reason QualityReport.disclosure exists at all.

    A family whose likeness is largely generated must be told at the moment
    they see it, in a sentence, not in terms nobody reads.
    """
    await add_photos(db, photo_set, 6)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    result = await finished(service, db, started.job_id, photo_set.owner_id)

    assert "generated rather than photographed" in result["disclosure"]
    assert result["measured_fraction"] == 0.25
    assert result["generated_fraction"] == 0.75


async def test_a_reconstructed_likeness_says_nothing_about_the_face_was_invented(
    db, store, sessions, photo_set
):
    await add_clip(store, photo_set, seconds=20.0)
    await add_photos(db, photo_set, 30, frames=True)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    result = await finished(service, db, started.job_id, photo_set.owner_id)

    assert "Nothing about the face was invented" in result["disclosure"]
    assert result["measured_fraction"] == 1.0
    assert result["generated_fraction"] == 0.0


async def test_the_disclosure_is_stored_on_the_avatar_not_only_returned_once(
    db, store, sessions, photo_set, engine
):
    """A page opened a week later still shows how much was generated."""
    await add_photos(db, photo_set, 10)

    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    await service.wait(started.job_id, 5.0)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as fresh:
        avatar = await fresh.get(Avatar, photo_set.avatar_id)
        assert avatar.splat_disclosure
        assert "generated rather than photographed" in avatar.splat_disclosure
        assert avatar.splat_measured_fraction == pytest.approx(0.42)
        assert avatar.splat_route == "generate"
        assert avatar.splat_key.startswith(f"tenants/{photo_set.owner_id}/")
        assert avatar.splat_gaussians > 0


async def test_an_unbuilt_avatar_has_no_measured_fraction_rather_than_a_zero_one(
    db, photo_set
):
    """NULL is "no likeness yet"; 0.0 would claim none of it was measured."""
    avatar = await db.get(Avatar, photo_set.avatar_id)
    assert avatar.splat_measured_fraction is None
    assert avatar.splat_disclosure is None


# --- a build is a job somebody can be told about --------------------------


async def test_progress_is_reportable_while_a_build_runs_and_after_it_finishes(
    db, store, sessions, photo_set
):
    release = asyncio.Event()

    class HeldBackend(FakeSplatBackend):
        """Finishes only when told to, so the middle of a build is observable."""

        async def collect(self, external_id, job, *, wait_s):
            await release.wait()
            return await super().collect(external_id, job, wait_s=wait_s)

    await add_photos(db, photo_set, 12)
    service = service_for(store, sessions, backend=HeldBackend(store))
    started = await service.start(db, photo_set.id, photo_set.owner_id)

    running = await service.read(db, started.job_id, photo_set.owner_id)
    assert running["status"] == "running"
    assert 0.0 < running["progress"] < 1.0
    # Nothing is claimed about the result while there is no result.
    assert running["disclosure"] is None

    release.set()
    done = await finished(service, db, started.job_id, photo_set.owner_id)
    assert done["status"] == "succeeded"
    assert done["progress"] == 1.0
    assert done["disclosure"]


async def test_a_failed_build_leaves_a_row_explaining_why_and_nothing_pending(
    db, store, sessions, photo_set
):
    await add_photos(db, photo_set, 12)
    builder = SplatBuilder(FakeSplatBackend(store, fail_in="collect"))
    service = SplatService(builder, store, sessions)

    started = await service.start(db, photo_set.id, photo_set.owner_id)
    result = await finished(service, db, started.job_id, photo_set.owner_id)

    assert result["status"] == "failed"
    assert "fail at collect" in result["error"]
    # The invariant inherited from the builder: a splat build is minutes of
    # GPU, and every exit from one leaves nothing running.
    assert builder.pending == ()
    assert service.running == ()


async def test_a_failed_build_leaves_the_avatar_without_a_likeness_or_a_claim(
    db, store, sessions, photo_set, engine
):
    await add_photos(db, photo_set, 12)
    service = service_for(store, sessions, backend=FakeSplatBackend(store, fail_in="submit"))
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    await service.wait(started.job_id, 5.0)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as fresh:
        avatar = await fresh.get(Avatar, photo_set.avatar_id)
        assert avatar.splat_key is None
        assert avatar.splat_disclosure is None


async def test_a_build_whose_runner_disappeared_is_failed_rather_than_building_forever(
    db, store, sessions, photo_set
):
    """A gateway restarted mid-build must not leave a family watching a bar."""
    lost = TrainingJob(
        owner_id=photo_set.owner_id,
        photo_set_id=photo_set.id,
        status=TrainingStatus.RUNNING,
        provider=f"{PROVIDER_PREFIX}:fake",
        started_at=datetime.now(UTC) - timedelta(seconds=Quality.STANDARD.wait_s + 600),
    )
    db.add(lost)
    await db.commit()

    service = service_for(store, sessions)
    result = await service.read(db, lost.id, photo_set.owner_id)

    assert result["status"] == "failed"
    assert "stopped reporting" in result["error"]
    assert "start it again" in result["error"]


async def test_a_build_still_within_its_own_timeout_is_left_alone(
    db, store, sessions, photo_set
):
    recent = TrainingJob(
        owner_id=photo_set.owner_id,
        photo_set_id=photo_set.id,
        status=TrainingStatus.RUNNING,
        provider=f"{PROVIDER_PREFIX}:fake",
        started_at=datetime.now(UTC) - timedelta(seconds=30),
    )
    db.add(recent)
    await db.commit()

    service = service_for(store, sessions)
    assert (await service.read(db, recent.id, photo_set.owner_id))["status"] == "running"


async def test_cancelling_a_build_stops_it_and_leaves_nothing_running(
    db, store, sessions, photo_set
):
    release = asyncio.Event()

    class HeldBackend(FakeSplatBackend):
        async def collect(self, external_id, job, *, wait_s):
            await release.wait()
            return await super().collect(external_id, job, wait_s=wait_s)

    await add_photos(db, photo_set, 12)
    backend = HeldBackend(store)
    builder = SplatBuilder(backend)
    service = SplatService(builder, store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)

    cancelled = await service.cancel(db, started.job_id, photo_set.owner_id)
    await asyncio.sleep(0)

    assert cancelled["status"] == "cancelled"
    assert service.running == ()
    assert builder.pending == ()
    # The backend was told, rather than the task merely being dropped.
    assert backend.cancelled


async def test_a_second_build_of_the_same_photo_set_is_refused_while_one_runs(
    db, store, sessions, photo_set
):
    """A splat build is minutes of GPU on one person; twice is the same
    likeness bought twice."""
    release = asyncio.Event()

    class HeldBackend(FakeSplatBackend):
        async def collect(self, external_id, job, *, wait_s):
            await release.wait()
            return await super().collect(external_id, job, wait_s=wait_s)

    await add_photos(db, photo_set, 12)
    service = service_for(store, sessions, backend=HeldBackend(store))
    await service.start(db, photo_set.id, photo_set.owner_id)

    with pytest.raises(SplatUnavailable):
        await service.start(db, photo_set.id, photo_set.owner_id)

    release.set()


async def test_a_photo_set_with_no_avatar_cannot_start_a_build(db, store, sessions, owner):
    orphan = PhotoSet(owner_id=owner.id, status=PhotoSetStatus.READY)
    db.add(orphan)
    await db.commit()
    await add_photos(db, orphan, 12)

    service = service_for(store, sessions)
    with pytest.raises(SplatUnavailable):
        await service.start(db, orphan.id, owner.id)


# --- tenant isolation -----------------------------------------------------


async def test_one_tenant_cannot_start_a_build_on_anothers_photo_set(
    db, store, sessions, photo_set, other_owner
):
    await add_photos(db, photo_set, 12)
    service = service_for(store, sessions)

    with pytest.raises(TenantError):
        await service.start(db, photo_set.id, other_owner.id)


async def test_one_tenant_cannot_read_or_cancel_anothers_build(
    db, store, sessions, photo_set, other_owner
):
    await add_photos(db, photo_set, 12)
    service = service_for(store, sessions)
    started = await service.start(db, photo_set.id, photo_set.owner_id)
    await service.wait(started.job_id, 5.0)

    with pytest.raises(TenantError):
        await service.read(db, started.job_id, other_owner.id)
    with pytest.raises(TenantError):
        await service.cancel(db, started.job_id, other_owner.id)


async def test_an_identity_training_run_is_not_readable_as_a_splat_build(
    db, store, sessions, photo_set
):
    """The two share a table; they must not share a status page.

    Reading a training run through the splat endpoints would report a LoRA as
    a likeness and, worse, let the splat status writer overwrite its row.
    """
    training = TrainingJob(
        owner_id=photo_set.owner_id,
        photo_set_id=photo_set.id,
        status=TrainingStatus.RUNNING,
        provider="local",
    )
    db.add(training)
    await db.commit()

    service = service_for(store, sessions)
    with pytest.raises(TenantError):
        await service.read(db, training.id, photo_set.owner_id)


# --- a route we have not shipped is refused, not attempted -----------------


@pytest.fixture
def no_gpu_may_be_reached(monkeypatch):
    """Reaching RunPod at all fails the test it happens in.

    Sharper than counting job rows: a refusal that has already opened a client
    or submitted a payload has already cost something, whatever it does next.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a refused build must never reach the GPU provider")

    monkeypatch.setattr("avatar.splat.build.ServerlessClient", refuse)


def runpod_service(store, sessions, *, reconstruct="reconstruct-endpoint", generate=""):
    backend = RunPodSplatBackend("key", reconstruct, generate)
    return SplatService(SplatBuilder(backend), store, sessions)


async def test_a_photographs_only_set_is_refused_while_that_route_has_no_worker(
    db, store, sessions, photo_set, no_gpu_may_be_reached
):
    await add_photos(db, photo_set, 20)
    service = runpod_service(store, sessions)

    with pytest.raises(SplatRefused) as refusal:
        await service.start(db, photo_set.id, photo_set.owner_id)

    assert refusal.value.decision.route is Route.REFUSE
    assert "only build a likeness from video" in refusal.value.decision.reasoning


async def test_that_refusal_leaves_no_job_row_and_nothing_running(
    db, store, sessions, photo_set, no_gpu_may_be_reached
):
    """Same posture as a refusal for thin material: nothing to explain later."""
    await add_photos(db, photo_set, 20)
    service = runpod_service(store, sessions)

    with pytest.raises(SplatRefused):
        await service.start(db, photo_set.id, photo_set.owner_id)

    assert service.running == ()
    jobs = (await db.execute(TrainingJob.__table__.select())).all()
    assert jobs == []


@pytest.fixture
def worker_reached(monkeypatch):
    """A RunPod endpoint that completes whatever it is given, remembering which.

    Patched over the class the backend constructs so the whole path is real -
    route chosen, endpoint picked, payload submitted, result parsed, row and
    disclosure written - with nothing rented.
    """
    reached: list[str] = []

    class Endpoint:
        def __init__(self):
            self._payload: dict = {}

        def submit(self, payload: dict) -> str:
            self._payload = payload
            return "runpod-000001"

        def status(self, job_id: str) -> JobResult:
            return JobResult(
                id=job_id,
                state=JobState.COMPLETED,
                output={"splat_key": self._payload["output_key"], "gaussians": 500_000},
                execution_ms=420_000,
            )

        def cancel(self, job_id: str) -> None:
            pass

    def make(api_key: str, endpoint_id: str) -> Endpoint:
        reached.append(endpoint_id)
        return Endpoint()

    monkeypatch.setattr("avatar.splat.build.ServerlessClient", make)
    return reached


async def test_a_video_set_builds_on_the_reconstruct_endpoint_that_exists(
    db, store, sessions, photo_set, worker_reached
):
    """One unshipped worker must not take the route that does exist with it."""
    await add_clip(store, photo_set, seconds=12.0)
    await add_photos(db, photo_set, 20, frames=True)
    service = runpod_service(store, sessions)

    started = await service.start(db, photo_set.id, photo_set.owner_id)
    result = await finished(service, db, started.job_id, photo_set.owner_id)

    assert started.decision.route is Route.RECONSTRUCT
    assert result["status"] == "succeeded"
    assert result["measured_fraction"] == 1.0
    assert worker_reached == ["reconstruct-endpoint"]


def test_the_runpod_builder_is_wired_to_one_endpoint_per_route(tmp_path):
    """What ships today: reconstruction has a verified worker, generation does not."""
    cfg = Settings(splat_backend="runpod", runpod_api_key="key")
    builder = build_splat_builder(cfg, LocalBlobStore(tmp_path))

    assert builder.backend_name == "runpod"
    assert builder.supports(Route.RECONSTRUCT)
    assert not builder.supports(Route.GENERATE)


def test_the_splat_endpoints_are_not_the_endpoint_that_animates_a_photograph():
    """runpod_endpoint_id is LivePortrait, and a splat payload means nothing to it."""
    cfg = Settings(runpod_endpoint_id="liveportrait-endpoint")

    assert cfg.runpod_splat_reconstruct_endpoint_id != cfg.runpod_endpoint_id
    assert cfg.runpod_splat_reconstruct_endpoint_id, "the verified splat worker"


# --- the refusal reaches the API in the shape the web app already reads ----


class RefusingService:
    """A service that only refuses, standing in at the HTTP edge."""

    def __init__(self, exc: SplatRefused):
        self._exc = exc

    async def start(self, db, photo_set_id, owner_id):
        raise self._exc


async def refusal_over_http(exc: SplatRefused) -> tuple[int, dict]:
    app = FastAPI()
    app.include_router(build_router(lambda: "tenant-a", lambda: None, RefusingService(exc)))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={REQUIRED_HEADER: REQUIRED_VALUE},
    ) as client:
        response = await client.post("/api/photo-sets/set-1/splat")
    return response.status_code, response.json()


def an_intake(photographs: int) -> Intake:
    return Intake(
        tenant_id="tenant-a",
        photo_set_id="set-1",
        photo_keys=tuple(
            photo_key("tenant-a", "set-1", f"photo-{i}.jpg") for i in range(photographs)
        ),
    )


async def test_an_unshipped_route_answers_in_the_same_shape_as_thin_material():
    """So the web app needs no change to show it, and no family meets a 500."""
    thin = SplatRefused(choose_route(an_intake(1)))
    unshipped = SplatRefused(route_unavailable(choose_route(an_intake(20))))

    thin_status, thin_body = await refusal_over_http(thin)
    unshipped_status, unshipped_body = await refusal_over_http(unshipped)

    assert thin_status == unshipped_status == 200
    assert thin_body.keys() == unshipped_body.keys()
    assert unshipped_body["status"] == "refused"
    assert unshipped_body["buildable"] is False


async def test_the_guidance_a_photographs_only_family_reads_is_one_plain_sentence():
    unshipped = SplatRefused(route_unavailable(choose_route(an_intake(20))))

    _, body = await refusal_over_http(unshipped)

    assert "at least 8 seconds" in body["guidance"]
    assert "face is visible" in body["guidance"]
    # And the trail support reads when a family asks why survives the trip.
    assert any("photographs accepted: 20" in line for line in body["considered"])
