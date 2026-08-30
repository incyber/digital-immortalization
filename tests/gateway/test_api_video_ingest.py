"""Uploading a clip, over HTTP.

This suite exists because of an outage. The video endpoint used to read the
whole upload into memory and then decode sixty frames and run face detection,
sharpness analysis and an object store upload on every one of them, inside the
request handler, on the event loop. Every other request in the process stalled,
the platform health check on the same port failed, and everybody on the site -
not only the person uploading - got a 502.

So the behaviour defended here is not "frames come out of a video". That was
never broken. It is that the upload returns immediately, that the work is
watchable while it happens, that a job whose process died says so, and above
all that the gateway keeps answering other requests the entire time.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from avatar.gateway.app import create_app
from avatar.gateway.csrf import REQUIRED_HEADER, REQUIRED_VALUE
from avatar.gateway.models import Base, Photo, TrainingStatus, VideoIngestJob
from avatar.storage.keys import source_clip_key


def a_clip(path, seconds: float = 20.0, fps: int = 15, size=(1280, 720)) -> bytes:
    """A real, decodable clip of a realistic size.

    Real rather than a few bytes with an .mp4 name, and 720p rather than
    thumbnail-sized, because what this suite measures is the cost of decoding
    and checking frames. A clip small enough to be free would defend nothing.
    """
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available")
    base = np.zeros((height, width, 3), np.uint8)
    base[:, :, 0] = np.linspace(20, 200, width, dtype=np.uint8)[None, :]
    base[:, :, 1] = np.linspace(200, 20, height, dtype=np.uint8)[:, None]
    for i in range(int(seconds * fps)):
        frame = base.copy()
        cv2.circle(frame, (width // 2 + (i % 40) - 20, height // 2), 120, (210, 180, 160), -1)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def an_image(w=1024, h=1024) -> bytes:
    rng = np.random.default_rng(2)
    frame = (np.full((h, w, 3), 140, np.uint8) + rng.integers(-30, 30, (h, w, 3))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    cfg.storage_root = str(tmp_path / "blobs")
    # Uploads spool here on their way to being read. Pointed at the test's own
    # directory so a suite run leaves nothing behind in the repository.
    cfg.assets_dir = str(tmp_path / "assets")
    cfg.splat_backend = "fake"

    engine = create_async_engine(cfg.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db_module._engine = engine
    db_module._factory = async_sessionmaker(engine, expire_on_commit=False)

    async with AsyncClient(
        transport=ASGITransport(app=create_app(cfg)),
        base_url="http://test",
        headers={REQUIRED_HEADER: REQUIRED_VALUE},
    ) as c:
        yield c
    await engine.dispose()


async def sign_in(client, email="a@example.com"):
    await client.post(
        "/api/auth/register", json={"email": email, "password": "a-long-enough-password"}
    )


async def a_set(client) -> str:
    return (await client.post("/api/photo-sets")).json()["id"]


async def upload(client, set_id: str, clip: bytes, name="clip.mp4"):
    return await client.post(
        f"/api/photo-sets/{set_id}/video", files={"file": (name, clip, "video/mp4")}
    )


async def finish(client, job_id: str, timeout_s: float = 90.0) -> dict:
    """Poll until the job is no longer running, and hand back the last read."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = (await client.get(f"/api/video-jobs/{job_id}")).json()
        if job["status"] not in ("queued", "running"):
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"video job {job_id} never finished")


async def test_the_upload_returns_before_any_frame_has_been_read(client, tmp_path):
    """The endpoint's cost is the size of the upload, not the length of the clip.

    This is the whole fix. The handler that took six seconds over an ordinary
    31-second recording is the handler that failed the platform health check.
    """
    await sign_in(client)
    set_id = await a_set(client)
    clip = a_clip(tmp_path / "clip.mp4")

    started = time.perf_counter()
    response = await upload(client, set_id, clip)
    elapsed = time.perf_counter() - started

    assert response.status_code == 202
    assert response.json()["job_id"]
    assert elapsed < 1.0, f"the upload took {elapsed:.2f}s before answering"

    # And it answered before the work was done, rather than merely being fast.
    job = (await client.get(f"/api/video-jobs/{response.json()['job_id']}")).json()
    assert job["frames_examined"] < job["frames_planned"] or job["status"] == "running"

    await finish(client, response.json()["job_id"])


async def test_the_gateway_keeps_answering_while_a_clip_is_being_read(client, tmp_path):
    """The test that actually defends the outage.

    A stalled event loop is invisible to a timer around a request: the request
    that is blocking still returns, eventually. What it stops is everything
    else. So this measures the gap between successive heartbeats - the time
    the loop had nothing to give anybody - across the whole of a job.

    The bound is a share of the window rather than a number of seconds,
    because a number of seconds only means anything on the machine it was
    measured on. Blocking, the loop was unavailable for about 80% of the job;
    off the loop it is about 10%, and that 10% is the heartbeat's own work.
    """
    await sign_in(client)
    set_id = await a_set(client)
    clip = a_clip(tmp_path / "clip.mp4")

    gaps: list[float] = []
    stop = asyncio.Event()
    beat_s = 0.01

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            assert (await client.get("/health")).status_code == 200
            await asyncio.sleep(beat_s)
            now = time.perf_counter()
            gaps.append(now - last - beat_s)
            last = now

    beat = asyncio.create_task(heartbeat())
    try:
        job_id = (await upload(client, set_id, clip)).json()["job_id"]
        job = await finish(client, job_id)
        assert job["status"] == "succeeded"
    finally:
        stop.set()
        await beat

    assert len(gaps) > 20, "the heartbeat never got to run"
    window = sum(gaps) + len(gaps) * beat_s
    unavailable = sum(gaps) / window
    assert unavailable < 0.35, (
        f"the loop was unavailable for {unavailable:.0%} of the job"
    )


async def test_no_frame_check_ever_runs_on_the_event_loop(client, tmp_path, monkeypatch):
    """The structural half of the same requirement, with no clock in it.

    Timing says the loop was mostly free; this says why. Every decode and
    every per-image check ran on some thread that was not the one serving
    requests - which is the property, and the one a faster machine cannot
    accidentally satisfy.
    """
    import avatar.ingest.service as ingest_service
    from avatar.ingest.video import FrameReader

    loop_thread = threading.get_ident()
    ran_on: set[int] = set()

    real_inspect = ingest_service.inspect_photo
    real_read_at = FrameReader.read_at

    def watched_inspect(*args, **kwargs):
        ran_on.add(threading.get_ident())
        return real_inspect(*args, **kwargs)

    def watched_read_at(self, *args, **kwargs):
        ran_on.add(threading.get_ident())
        return real_read_at(self, *args, **kwargs)

    monkeypatch.setattr(ingest_service, "inspect_photo", watched_inspect)
    monkeypatch.setattr(FrameReader, "read_at", watched_read_at)

    await sign_in(client)
    set_id = await a_set(client)
    job = await finish(
        client,
        (await upload(client, set_id, a_clip(tmp_path / "clip.mp4", seconds=6.0))).json()[
            "job_id"
        ],
    )

    assert job["status"] == "succeeded"
    assert ran_on, "nothing was decoded or checked at all"
    assert loop_thread not in ran_on, "frame work ran on the event loop"


async def test_progress_is_readable_while_it_runs_and_after_it_finishes(client, tmp_path):
    """A count somebody can watch, not a spinner.

    The frame total comes from the clip's own metadata before any decoding, so
    the bar has a denominator from the first poll rather than only at the end.
    """
    await sign_in(client)
    set_id = await a_set(client)
    clip = a_clip(tmp_path / "clip.mp4")

    job_id = (await upload(client, set_id, clip)).json()["job_id"]

    seen: list[dict] = []
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        job = (await client.get(f"/api/video-jobs/{job_id}")).json()
        seen.append(job)
        if job["status"] not in ("queued", "running"):
            break
        await asyncio.sleep(0.02)

    mid = [j for j in seen if j["status"] == "running" and j["frames_planned"] > 0]
    assert mid, "no poll during the work reported a frame count"
    assert any(0 < j["frames_examined"] < j["frames_planned"] for j in mid), (
        "progress was never partway through"
    )
    assert all(0.0 <= j["progress"] < 1.0 for j in mid), "the bar reached the end early"

    # Progress only ever moves forward.
    examined = [j["frames_examined"] for j in mid]
    assert examined == sorted(examined)

    done = seen[-1]
    assert done["status"] == "succeeded"
    assert done["progress"] == 1.0
    assert done["frames_examined"] > 0
    assert done["frames_usable"] <= done["frames_examined"]


async def test_the_frames_land_in_the_photo_set_and_the_clip_is_kept(client, tmp_path):
    """What the job is for, checked once so the plumbing is not merely fast.

    The clip itself is kept as well as sampled: it is the base the lip-sync
    renderer drives, and the head moves in the result because it moved in the
    recording.
    """
    import avatar.gateway.db as db_module

    await sign_in(client)
    set_id = await a_set(client)
    owner_id = (await client.get("/api/me")).json()["id"]
    clip = a_clip(tmp_path / "clip.mp4", seconds=6.0)

    job = await finish(client, (await upload(client, set_id, clip)).json()["job_id"])
    assert job["status"] == "succeeded"

    async with db_module._factory() as db:
        photos = (
            (await db.execute(select(Photo).where(Photo.photo_set_id == set_id)))
            .scalars()
            .all()
        )
    assert len(photos) == job["frames_examined"]
    assert all(p.filename.startswith("frame-") for p in photos)

    stored = tmp_path / "blobs" / source_clip_key(owner_id, set_id)
    assert stored.exists(), "the source clip was not kept"

    # And the spooled upload is gone. Two hundred and fifty megabytes of these
    # accumulating on the volume is a full disk with nothing pointing at the
    # cause, so the job deletes its own file whatever the outcome.
    spool = tmp_path / "assets" / "uploads"
    assert not spool.exists() or not list(spool.glob("*"))


async def test_a_job_whose_process_died_is_reported_as_failed(client, tmp_path):
    """A restart must leave evidence, not a bar frozen at forty percent.

    The tasks live in memory and the rows live in the database, so a row still
    claiming to run, past the point the work would have finished, with no task
    behind it, is a gateway that restarted mid-job. Somebody has to be told.
    """
    import avatar.gateway.db as db_module

    await sign_in(client)
    set_id = await a_set(client)
    owner_id = (await client.get("/api/me")).json()["id"]

    # Written directly: this is the state a killed process leaves behind, and
    # there is no way to reach it by asking the gateway politely.
    async with db_module._factory() as db:
        job = VideoIngestJob(
            owner_id=owner_id,
            photo_set_id=set_id,
            status=TrainingStatus.RUNNING,
            filename="clip.mp4",
            frames_planned=60,
            frames_examined=24,
            started_at=datetime.now(UTC) - timedelta(hours=4),
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    body = (await client.get(f"/api/video-jobs/{job_id}")).json()
    assert body["status"] == "failed"
    assert "upload it again" in body["error"]
    assert body["progress"] == 0.0


async def test_a_job_still_within_its_deadline_is_left_alone(client, tmp_path):
    """The other half of it: a slow job is not a dead one.

    Without this the expiry rule would be free to fail everything, and the
    test above would still pass.
    """
    import avatar.gateway.db as db_module

    await sign_in(client)
    set_id = await a_set(client)
    owner_id = (await client.get("/api/me")).json()["id"]

    async with db_module._factory() as db:
        job = VideoIngestJob(
            owner_id=owner_id,
            photo_set_id=set_id,
            status=TrainingStatus.RUNNING,
            frames_planned=60,
            frames_examined=24,
            started_at=datetime.now(UTC) - timedelta(seconds=30),
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    assert (await client.get(f"/api/video-jobs/{job_id}")).json()["status"] == "running"


async def test_a_second_clip_is_refused_while_one_is_still_being_read(client, tmp_path):
    """Two at once would interleave frame numbers into one set and race on the
    stored clip, and the second upload would silently win."""
    await sign_in(client)
    set_id = await a_set(client)
    clip = a_clip(tmp_path / "clip.mp4")

    first = await upload(client, set_id, clip)
    assert first.status_code == 202

    second = await upload(client, set_id, clip, name="again.mp4")
    assert second.status_code == 409
    assert "already being read" in second.json()["detail"]

    await finish(client, first.json()["job_id"])


async def test_a_file_that_is_not_a_video_is_refused_before_it_is_stored(client, tmp_path):
    """Writing a quarter of a gigabyte to the volume and then refusing it is
    work done for nothing."""
    await sign_in(client)
    set_id = await a_set(client)

    response = await client.post(
        f"/api/photo-sets/{set_id}/video",
        files={"file": ("notes.txt", b"not a video at all", "text/plain")},
    )
    assert response.status_code == 400
    assert not list((tmp_path / "assets" / "uploads").glob("*")) if (
        tmp_path / "assets" / "uploads"
    ).exists() else True


async def test_a_clip_nothing_can_decode_fails_the_job_with_a_reason(client, tmp_path):
    """The refusal survives the request that started it.

    Before, an undecodable clip was a 400 to somebody still on the page. Now
    the upload has already been accepted, so the reason has to be written down
    where they will read it afterwards.
    """
    await sign_in(client)
    set_id = await a_set(client)

    accepted = await upload(client, set_id, b"MOV rubbish that is not a container")
    assert accepted.status_code == 202

    job = await finish(client, accepted.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]


async def test_another_tenant_cannot_read_a_video_job(client, tmp_path):
    await sign_in(client, "owner@example.com")
    set_id = await a_set(client)
    job_id = (await upload(client, set_id, a_clip(tmp_path / "clip.mp4", seconds=4.0))).json()[
        "job_id"
    ]
    await finish(client, job_id)

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")
    assert (await client.get(f"/api/video-jobs/{job_id}")).status_code == 404


async def test_another_tenant_cannot_upload_a_clip_into_a_photo_set(client, tmp_path):
    await sign_in(client, "owner@example.com")
    set_id = await a_set(client)
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await upload(client, set_id, a_clip(tmp_path / "clip.mp4", seconds=4.0))
    assert response.status_code == 404
    # And the refusal left nothing on the volume.
    spool = tmp_path / "assets" / "uploads"
    assert not spool.exists() or not list(spool.glob("*"))


async def test_checking_one_photograph_does_not_run_on_the_event_loop(
    client, monkeypatch
):
    """The photo endpoint is the same bug in slow motion.

    One photograph is one face detection, which is fast enough that the
    request can honestly wait for its own verdict - and it should, because
    somebody choosing files needs to be told which ones were unusable while
    they still have the folder open. Twenty-five of them in a row on the event
    loop, though, is the gateway unavailable twenty-five times.

    So the endpoint stayed inline and the check moved off the loop. What was
    wrong was where it ran, not that the caller waits for it.
    """
    import avatar.ingest.service as ingest_service

    loop_thread = threading.get_ident()
    ran_on: set[int] = set()
    real_inspect = ingest_service.inspect_photo

    def watched(*args, **kwargs):
        ran_on.add(threading.get_ident())
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(ingest_service, "inspect_photo", watched)

    await sign_in(client)
    set_id = await a_set(client)
    response = await client.post(
        f"/api/photo-sets/{set_id}/photos",
        files={"file": ("p.jpg", an_image(), "image/jpeg")},
    )

    assert response.status_code == 201
    # The verdict is still in this response: the customer is told about this
    # photograph now, not after twenty-four more.
    assert "accepted" in response.json()
    assert ran_on and loop_thread not in ran_on, "the check ran on the event loop"
