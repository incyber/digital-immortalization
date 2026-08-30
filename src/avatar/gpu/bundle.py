"""Packaging the private half of a GPU worker.

The published images hold open-source models and a bootstrap that can only
download, verify and exec. Everything that is ours - the handler, the renderer
service, the licence-clean detector, the lease supervisor - travels in an
archive that lives in the customer's own object storage and never enters a
container registry.

The hash returned here is the security boundary, not the bucket. Storage
credentials sit in the worker's environment, so a bucket alone would let anyone
holding them swap the code a GPU runs. The endpoint is configured with the
digest, the bootstrap refuses to execute anything else, and a missing digest is
treated as a mismatch rather than as permission.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# What each worker needs, and nothing else. Listed explicitly rather than
# globbed: an archive built from a directory listing eventually picks up a
# stray .env or a notebook, and this one is uploaded to storage a worker reads.
SERVERLESS_FILES = {
    "handler.py": "infra/serverless/handler.py",
    "facegeom/server.py": "infra/facegeom/server.py",
    "facegeom/facegeom_core.py": "infra/facegeom/facegeom_core.py",
    "liveportrait/facegeom_shim.py": "infra/liveportrait/facegeom_shim.py",
    "liveportrait/patch_cropper.py": "infra/liveportrait/patch_cropper.py",
}

MUSETALK_FILES = {
    "handler.py": "infra/lease/supervisor.py",
    "server.py": "infra/musetalk/server.py",
    "bbox.py": "infra/musetalk/bbox.py",
    "facegeom/server.py": "infra/facegeom/server.py",
    "facegeom/facegeom_core.py": "infra/facegeom/facegeom_core.py",
}

# The splat worker. Flat rather than nested because the bootstrap execs
# handler.py from the top of APP_DIR, which puts that directory on sys.path -
# so reconstruct.py and generate.py import each other by plain name, exactly as
# they do in the repository.
SPLAT_FILES = {
    "handler.py": "infra/splatworker/handler.py",
    "reconstruct.py": "infra/splatworker/reconstruct.py",
    "generate.py": "infra/splatworker/generate.py",
}

# A shell on a GPU. Carries the worker's own modules alongside the agent so
# that a debugging session can import and call the real reconstruct code rather
# than a copy of it - the whole point is to run what production runs.
DEBUG_FILES = {
    "handler.py": "infra/splatworker/agent.py",
    "reconstruct.py": "infra/splatworker/reconstruct.py",
    "generate.py": "infra/splatworker/generate.py",
    "worker.py": "infra/splatworker/handler.py",
}

BUNDLES = {
    "serverless": SERVERLESS_FILES,
    "musetalk": MUSETALK_FILES,
    "splat": SPLAT_FILES,
    "debug": DEBUG_FILES,
}


@dataclass(frozen=True)
class Bundle:
    name: str
    data: bytes
    sha256: str

    @property
    def key(self) -> str:
        """Where it goes in the bucket, named by content.

        Content-addressed so a new bundle never overwrites the one a running
        endpoint is configured against. Rolling back is then a matter of
        pointing at the previous digest rather than rebuilding anything.
        """
        return f"gpu-bundles/{self.name}-{self.sha256[:16]}.tar.gz"


def build(name: str, root: Path) -> Bundle:
    """Pack one worker's private files into a reproducible archive."""
    files = BUNDLES.get(name)
    if files is None:
        raise ValueError(f"unknown bundle {name!r}; expected one of {sorted(BUNDLES)}")

    buffer = io.BytesIO()
    # The gzip layer is built explicitly with mtime=0. tarfile's "w:gz" writes
    # the current time into the gzip header, so two builds of identical sources
    # a second apart produced different bytes and a different digest - which
    # the reproducibility test caught only intermittently, because inside one
    # second it passes. A digest that drifts on its own is worse than no digest
    # at all: the endpoint is pinned to one, and the worker refuses anything
    # else.
    gz = gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0)
    with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for arcname, relative in sorted(files.items()):
            source = root / relative
            if not source.exists():
                raise FileNotFoundError(f"{source} is missing; the bundle would be incomplete")
            info = tarfile.TarInfo(arcname)
            payload = source.read_bytes()
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(payload))

    gz.close()
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    logger.info(f"bundle {name}: {len(files)} files, {len(data) // 1024}KB, {digest[:16]}...")
    return Bundle(name=name, data=data, sha256=digest)
