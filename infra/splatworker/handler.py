"""RunPod Serverless worker: a person's photographs or video in, their splat out.

Its own image rather than a task on the existing worker. The other one carries
LivePortrait and CosyVoice and is already 15GB; this one carries gsplat and
TRELLIS, has a different CUDA toolchain requirement (gsplat compiles kernels),
and runs for minutes rather than seconds. Sharing an image would mean every
sixty-second voice job paying the cold start of a 3D generator it never calls.

WHAT THIS FILE IS RESPONSIBLE FOR, AND WHAT IT DELIBERATELY IS NOT
=================================================================
It is responsible for: dispatching on the route the orchestrator already
decided, fetching source assets *by key* from the tenant's own prefix,
producing exactly one .splat, writing it back, and reporting counts.

It is *not* responsible for deciding how much of the likeness is measured, and
it must not be able to influence that. avatar/splat/build.py derives the
measured fraction from the route alone - RECONSTRUCT is 1.0, GENERATE is capped
at 0.6 by MAX_MEASURED_ON_GENERATION - so there is no field in this worker's
output where a generated build could claim to have been photographed. That is
enforced here as well as there: _result() emits a fixed set of names and
nothing else, so a future edit cannot add one by accident.

`angular_coverage` is the one number the worker does contribute, and it cannot
launder anything: build.py feeds it into QualityReport.angular_coverage, which
is a description of the viewing sphere and is explicitly not a quality score.
The measured fraction does not read it.

KEYS, NEVER BYTES
=================
Source assets arrive as storage keys and are fetched here. A thirty-image set
is tens of megabytes; carrying it in the job would put every family
photograph in the queue, in the retry, and in every log line that prints a
payload. SplatJob refuses to construct with bytes, and this end refuses to
accept them, because the two ends of a contract that only one end enforces is
a contract that will eventually be broken by the other.

Every key is also re-checked against the tenant's prefix. The orchestrator
checks it too. This is the last place before a read, and the failure it
prevents - one family's build reading another family's photographs - is not
one worth trusting a single check with.

IMPORT-TIME SAFETY
==================
This module imports on a machine with no CUDA, no gsplat and no weights, the
same as infra/serverless/handler.py does: torch, gsplat, TRELLIS, mediapipe,
cv2 and boto3 are all imported inside the functions that need them, and the
health task reports what is absent instead of dying. A worker that cannot
reach its weights should return a usable error, not crash-loop.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from pathlib import Path


def _fatal(detail: str) -> None:
    """Put a traceback somewhere it can be read.

    Defined before the first application import on purpose. The platform's
    worker logs are unreachable from the API, so a module that dies while
    importing leaves no trace at all: the bootstrap's last breadcrumb says
    "exec handler.py" and then nothing, forever, which is what happened.

    Best effort. A worker that cannot report why it failed must still fail.
    """
    print(detail, flush=True)
    try:
        import sys as _sys

        _sys.path.insert(0, "/opt")
        import bootstrap

        bootstrap.log(detail)
    except Exception:  # noqa: BLE001, S110 - reporting must never mask the fault
        pass


def _ship_stdout() -> None:
    """Mirror this process's output to the operator's bucket, continuously.

    The breadcrumbs were enough to see how far the bootstrap got and no
    further: each one builds its own client and takes about a second, so a
    worker being stopped mid-sentence loses the sentence. Meanwhile the thing
    worth reading - what the RunPod SDK itself says when it starts, polls, and
    stops - was going to a log the API does not expose.

    So stdout and stderr are teed into a buffer and flushed every two seconds
    by a daemon thread that holds one client open. Best effort throughout: if
    the bucket is unreachable the worker still runs, and the thread never
    delays an exit.
    """
    bucket = os.environ.get("BUNDLE_BUCKET") or S3_BUCKET
    if not bucket:
        return

    import io
    import sys
    import threading

    pod = os.environ.get("RUNPOD_POD_ID", "unknown")
    pending: list[str] = []
    lock = threading.Lock()

    class _Tee(io.TextIOBase):
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def write(self, text: str) -> int:
            with lock:
                pending.append(text)
            return self._wrapped.write(text)

        def flush(self) -> None:
            self._wrapped.flush()

        # Anything a library reaches for on a real stream, passed straight
        # through. A tee that answers "no such attribute" to .buffer or
        # .fileno() would break the very code it is meant to observe.
        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

        # Defined explicitly rather than left to __getattr__: TextIOBase
        # already provides these and raises from them, so a missing-attribute
        # hook never sees the call.
        def fileno(self) -> int:
            return self._wrapped.fileno()

        def isatty(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

    sys.stdout = _Tee(sys.stdout)
    sys.stderr = _Tee(sys.stderr)

    def pump() -> None:
        import time

        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("BUNDLE_ENDPOINT_URL") or S3_ENDPOINT_URL or None,
            aws_access_key_id=os.environ.get("BUNDLE_ACCESS_KEY") or S3_ACCESS_KEY,
            aws_secret_access_key=os.environ.get("BUNDLE_SECRET_KEY") or S3_SECRET_KEY,
            region_name="auto",
            config=Config(connect_timeout=5, read_timeout=10),
        )
        sequence = 0
        while True:
            time.sleep(2)
            with lock:
                text, pending[:] = "".join(pending), []
            if not text:
                continue
            sequence += 1
            try:
                client.put_object(
                    Bucket=bucket,
                    Key=f"worker-out/{pod}/{sequence:04d}.txt",
                    Body=text.encode()[:200_000],
                    ContentType="text/plain",
                )
            except Exception:  # noqa: BLE001, S110 - reporting must not stop work
                pass

    threading.Thread(target=pump, daemon=True).start()


try:
    import generate
    import reconstruct
except BaseException as exc:  # re-raised after reporting
    import traceback

    _fatal(f"handler failed to import: {exc}\n{traceback.format_exc()}")
    raise

# Object storage, named exactly as it is in avatar/config.py so one set of
# variables configures the gateway and the worker.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

# The prefix every key must sit under, mirroring avatar/storage/keys.py. The
# trailing slash is load-bearing there and here: without it "tenants/abc"
# prefix-matches "tenants/abcd/secret.jpg" and a build crosses tenants.
KEY_ROOT = "tenants"
SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# Ceiling on one downloaded asset and on the set. A video longer than a few
# minutes is not more coverage of a head, it is more of the same head, and the
# worker should say so rather than spend fifteen minutes proving it.
MAX_ASSET_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024

# Most photographs a generated build will look at. Beyond this the appearance
# correction is re-seeing the same face and the marginal photograph costs a
# landmark pass for nothing.
MAX_PHOTOS = 40

ROUTES = ("reconstruct", "generate")

# Exactly what this worker may say about a finished build. Named as a constant
# and applied in _result() so that the guarantee - no field here can report a
# generated splat as measured - survives future edits to this file.
RESULT_FIELDS = (
    "splat_key", "gaussians", "bytes", "angular_coverage",
    "route", "views_used", "notes", "seconds",
)


class TaskError(RuntimeError):
    """A failure that should be reported to the caller, not retried."""


# --------------------------------------------------------------------------
# payload


def _text(field: str, value: object) -> str:
    """A string, and specifically not image data wearing a string's clothes."""
    if isinstance(value, bytes | bytearray | memoryview):
        raise TaskError(
            f"{field} arrived as bytes; source assets travel as storage keys so "
            "that a family's photographs do not pass through the job queue"
        )
    if not isinstance(value, str) or not value:
        raise TaskError(f"{field} must be a non-empty string")
    return value


def _tenant(payload: dict) -> str:
    tenant_id = _text("tenant_id", payload.get("tenant_id"))
    if not SAFE_ID.match(tenant_id):
        raise TaskError(f"tenant_id {tenant_id!r} is not a valid identifier")
    return tenant_id


def _key(field: str, value: object, tenant_id: str) -> str:
    """One storage key, confined to this tenant's prefix."""
    key = _text(field, value)
    if not key.startswith(f"{KEY_ROOT}/{tenant_id}/"):
        raise TaskError(f"{field} {key!r} is outside tenant {tenant_id}'s prefix")
    if ".." in key:
        raise TaskError(f"{field} {key!r} contains a path traversal")
    return key


def _positive(field: str, value: object) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TaskError(f"{field} must be a whole number; got {value!r}") from exc
    if number <= 0:
        raise TaskError(f"{field} must be greater than zero; got {number}")
    return number


# --------------------------------------------------------------------------
# storage


def _client():
    """The S3-compatible client, built once a job actually needs it.

    Lazy so that importing this module on a machine with no credentials - a
    laptop running the tests, for instance - neither fails nor silently
    constructs a client pointed at nothing.
    """
    if not S3_BUCKET:
        raise TaskError("S3_BUCKET is not set, so the worker has nowhere to read or write")

    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL or None,
        aws_access_key_id=S3_ACCESS_KEY or None,
        aws_secret_access_key=S3_SECRET_KEY or None,
        # SigV4 is required by R2 and by newer S3 regions, as in
        # avatar/storage/s3.py.
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
    )


def _download(client, keys: list[str], workdir: Path) -> list[Path]:
    """Fetch the source assets, named by position rather than by their key.

    Local names are index-based because a storage key is not a safe filename
    and sanitising one into a filename is how two different keys become the
    same file.
    """
    paths, total = [], 0
    for index, key in enumerate(keys):
        suffix = Path(key).suffix[:8] or ".bin"
        target = workdir / f"src-{index:03d}{suffix}"
        try:
            client.download_file(S3_BUCKET, key, str(target))
        except Exception as exc:
            # Deliberately wide: botocore raises a large family of errors and
            # the caller needs one sentence naming the key, not the taxonomy.
            raise TaskError(f"{key} could not be read: {type(exc).__name__}") from exc

        size = target.stat().st_size
        if size == 0:
            raise TaskError(f"{key} is empty")
        if size > MAX_ASSET_BYTES:
            raise TaskError(f"{key} is {size // (1024 * 1024)}MB, over the worker's limit")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise TaskError("the source material exceeds what one build may download")
        paths.append(target)
    return paths


def _upload(client, key: str, data: bytes) -> None:
    try:
        client.put_object(
            Bucket=S3_BUCKET, Key=key, Body=data,
            ContentType="application/octet-stream",
        )
    except Exception as exc:
        raise TaskError(
            f"the splat could not be written to {key}: {type(exc).__name__}"
        ) from exc


# --------------------------------------------------------------------------
# result


def _result(
    *, splat_key: str, cloud, size_bytes: int, route: str, seconds: float
) -> dict:
    """The only shape this worker may return.

    Assembled from RESULT_FIELDS rather than written inline, because the
    guarantee that matters is negative: there is no field here through which a
    generated build could be reported as measured. avatar/splat/build.py reads
    splat_key, gaussians, bytes and angular_coverage; the rest is for logs.
    """
    values = {
        "splat_key": splat_key,
        "gaussians": int(cloud.count),
        "bytes": int(size_bytes),
        "angular_coverage": float(cloud.angular_coverage),
        "route": route,
        "views_used": int(cloud.views_used),
        "notes": list(cloud.notes),
        "seconds": round(seconds, 2),
    }
    return {name: values[name] for name in RESULT_FIELDS}


# --------------------------------------------------------------------------
# the job


def _splat(payload: dict) -> dict:
    """One build. The route was decided before this worker was started.

    The worker does not re-run choose_route and must not: the decision is
    recorded on the job row and shown to the customer, and a worker that
    reached its own conclusion would produce a splat whose explanation belongs
    to a different build.

    It does refuse a route it should never have been sent. REFUSE is not a
    build; avatar/splat/build.py raises SplatRefused before submitting one, so
    a refusal arriving here means something bypassed the planner, and running
    it anyway would hand a family the likeness the system decided not to make.
    """
    route = _text("route", payload.get("route"))
    if route == "refuse":
        raise TaskError(
            "a refused decision is not a job: this material was judged unable to "
            "produce a likeness and no splat may be built from it"
        )
    if route not in ROUTES:
        raise TaskError(f"unknown route {route!r}; expected one of {list(ROUTES)}")

    tenant_id = _tenant(payload)
    output_key = _key("output_key", payload.get("output_key"), tenant_id)
    budget = _positive("gaussian_budget", payload.get("gaussian_budget"))
    iterations = _positive("iterations", payload.get("iterations"))

    raw_photos = payload.get("photo_keys") or []
    if not isinstance(raw_photos, list):
        raise TaskError("photo_keys must be a list of storage keys")
    if len(raw_photos) > MAX_PHOTOS:
        raise TaskError(f"more than {MAX_PHOTOS} photographs were supplied")
    photo_keys = [_key("photo_key", key, tenant_id) for key in raw_photos]

    # Every key is resolved before storage is touched, so a job naming a key
    # outside its tenant is told exactly that rather than being told about the
    # first unrelated thing the worker happens to trip over.
    if route == "reconstruct":
        sources = [_key("video_key", payload.get("video_key"), tenant_id)]
    else:
        anchor_key = _key("anchor_key", payload.get("anchor_key"), tenant_id)
        if anchor_key not in photo_keys:
            raise TaskError("the anchor photograph is not among the photographs supplied")
        # The anchor is downloaded first so its local path is unambiguous; the
        # rest follow in the order the family uploaded them.
        sources = [anchor_key] + [key for key in photo_keys if key != anchor_key]

    workdir = Path(tempfile.mkdtemp(prefix="splat-", dir="/tmp"))
    client = _client()
    started = time.perf_counter()
    paths = _download(client, sources, workdir)

    if route == "reconstruct":
        cloud = reconstruct.reconstruct(
            paths[0], iterations=iterations, gaussian_budget=budget
        )
    else:
        cloud = generate.generate(paths[0], paths[1:], gaussian_budget=budget)

    data = reconstruct.write_splat(cloud)
    if len(data) != cloud.count * reconstruct.BYTES_PER_GAUSSIAN:
        raise TaskError("the exported splat is not the size its Gaussian count implies")
    _upload(client, output_key, data)

    return _result(
        splat_key=output_key,
        cloud=cloud,
        size_bytes=len(data),
        route=route,
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# health


def _health(_payload: dict) -> dict:
    """What this worker can do, without doing any of it.

    Also the licence check. The existing worker records that InsightFace is
    absent rather than merely unused; the same reasoning applies to the Inria
    rasteriser and its derivatives, which TRELLIS's optional setup flags would
    otherwise install. If any of these ever appears in the image, it shows up
    in a health dump rather than in a licence audit.
    """
    import importlib.util

    def present(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception as exc:  # noqa: BLE001 - a worker with no torch must still answer
        cuda, gpu = False, f"torch unavailable: {type(exc).__name__}"

    return {
        "cuda": cuda,
        "gpu": gpu,
        "gsplat": present("gsplat"),
        "trellis": present("trellis"),
        "mediapipe": present("mediapipe"),
        "trellis_weights": Path(generate.TRELLIS_MODEL_DIR).exists(),
        "face_landmarker": Path(reconstruct.FACE_MODEL_PATH).exists(),
        "segmenter": Path(reconstruct.SEGMENTER_MODEL_PATH).exists(),
        "storage_configured": bool(S3_BUCKET),
        # Every one of these must be false. They are the implementations whose
        # licences forbid commercial use, and the image is built to not contain
        # them; this is where that is observable at runtime.
        "restricted_modules_present": sorted(
            name for name in (
                "diff_gaussian_rasterization", "diffoctreerast", "nvdiffrast",
                "insightface", "smplx",
            ) if present(name)
        ),
    }


TASKS = {"splat": _splat, "health": _health}


def handler(job: dict) -> dict:
    """Dispatch one job.

    Returns an {"error": ...} object rather than raising, exactly as the
    existing worker does: a malformed job is one failed build with a readable
    reason, not a worker the platform retries three times at GPU rates. The
    error text is a sentence because it reaches a support agent, and a
    traceback does not tell them which photograph was wrong.
    """
    payload = job.get("input") or {}
    _fatal(f"job claimed: {str(payload)[:200]}")
    if not isinstance(payload, dict):
        return {"error": "the job input must be an object"}

    name = payload.get("task", "splat")
    task = TASKS.get(name)
    if task is None:
        return {"error": f"unknown task {name!r}; expected one of {sorted(TASKS)}"}

    job_id = payload.get("job_id") or uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        result = task(payload)
    except (TaskError, reconstruct.ReconstructError, generate.GenerateError) as exc:
        return {"error": str(exc), "task": name, "job_id": job_id}
    except Exception as exc:  # noqa: BLE001
        # Unexpected failures are returned rather than propagated: a crash here
        # has the same billing cost and less information in it.
        return {"error": f"{type(exc).__name__}: {exc}", "task": name, "job_id": job_id}

    return {
        **result, "task": name, "job_id": job_id,
        "total_seconds": round(time.perf_counter() - started, 2),
    }


if __name__ == "__main__":
    # Imported here rather than at module scope so this file loads - and is
    # testable - on a machine that has no runpod package, which is every
    # machine except the worker.
    #
    # Wrapped, because the worker logs are unreachable from the API and a
    # handler that dies on import is indistinguishable from one that never
    # started. The traceback goes to the same place the bootstrap writes its
    # breadcrumbs, which is somewhere we can actually read.
    # A worker that dies without raising leaves nothing behind either, and
    # that is what was happening: the bootstrap said "exec handler.py", six
    # seconds passed, and the container started over. So the exit itself is
    # reported, whatever caused it, and a fault handler covers the crash that
    # never reaches Python at all.
    import atexit
    import faulthandler

    _ship_stdout()

    _crash = open("/tmp/handler-crash.txt", "w")  # noqa: SIM115 - lives for the process
    faulthandler.enable(file=_crash)

    def _report_exit() -> None:
        _crash.flush()
        try:
            detail = Path("/tmp/handler-crash.txt").read_text()[-3000:]
        except OSError:
            detail = "(no fault output)"
        _fatal(f"handler process is exiting\n{detail}")

    atexit.register(_report_exit)

    # Signals do not run atexit handlers, and the worker was being stopped
    # without any of the exit paths above reporting anything. A platform that
    # scales a worker down sends SIGTERM; a segmentation fault sends SIGSEGV.
    # Both look identical from the outside - the container simply starts over -
    # so each one says its own name before it goes.
    import signal

    def _on_signal(number, _frame):
        _fatal(f"handler received signal {number} ({signal.Signals(number).name})")
        raise SystemExit(128 + number)

    for _signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_signal, _on_signal)
        except (OSError, ValueError):  # not settable in this context
            pass

    try:
        _fatal("importing runpod")
        import runpod

        # Which mode the SDK is about to choose, said out loud. A worker that
        # cannot see the platform's job-taking variables starts a local test
        # server instead of polling, and from the outside that is
        # indistinguishable from a healthy idle worker - which is what it
        # looked like: one ready worker, two jobs queued, nothing claimed.
        seen = sorted(name for name in os.environ if name.startswith("RUNPOD_"))
        _fatal(
            f"runpod {getattr(runpod, '__version__', '?')}; "
            f"RUNPOD_ env present: {seen}; "
            f"argv={__import__('sys').argv}; starting serverless loop"
        )
        runpod.serverless.start({"handler": handler})
        _fatal("runpod.serverless.start returned, which it should not")
    except BaseException as exc:  # includes SystemExit, which is the quiet one
        import traceback

        _fatal(f"handler failed to start: {exc}\n{traceback.format_exc()}")
        raise
