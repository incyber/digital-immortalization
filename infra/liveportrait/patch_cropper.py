"""Rewrite LivePortrait's cropper to use the licence-clean detector.

Applied at image build time rather than as a runtime monkeypatch, so a version
of LivePortrait whose cropper has changed shape fails the build loudly instead
of silently falling back to InsightFace.

Only two edits are needed: the import, and the construction. Everything
downstream reads `landmark_2d_106`, which the shim provides.
"""

import pathlib
import sys

CROPPER = pathlib.Path("src/utils/cropper.py")

OLD_IMPORT = "from .face_analysis_diy import FaceAnalysisDIY"
NEW_IMPORT = (
    "# Replaced: LivePortrait's LICENSE requires InsightFace's detection\n"
    "# models to be removed for commercial use. See facegeom_shim.\n"
    "from .facegeom_shim import FaceAnalysisShim as FaceAnalysisDIY"
)


def main() -> int:
    source = CROPPER.read_text()

    if OLD_IMPORT not in source:
        print(f"FAILED: expected import not found in {CROPPER}.", file=sys.stderr)
        print("Upstream has changed shape; re-check the InsightFace call sites.", file=sys.stderr)
        return 1

    source = source.replace(OLD_IMPORT, NEW_IMPORT)
    CROPPER.write_text(source)

    # The shim ignores root/providers, but leaving them in place keeps the
    # diff minimal and the upstream call signature intact.
    remaining = [
        line for line in source.splitlines()
        if "insightface" in line.lower() and "LICENSE" not in line and "#" not in line
    ]
    if remaining:
        print("WARNING: insightface still referenced:", file=sys.stderr)
        for line in remaining:
            print("   ", line.strip(), file=sys.stderr)

    print("cropper patched to use the MediaPipe-backed detector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
