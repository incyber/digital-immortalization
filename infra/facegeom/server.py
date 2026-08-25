"""Face geometry over HTTP.

One endpoint. Give it an image, get back the geometry both renderers need:
a bounding box, a mouth rectangle derived from the lips rather than assumed,
landmarks, and a soft face mask.

This exists so that neither renderer has to ship its own detector. MuseTalk's
preprocessing wants mmpose/DWPose and LivePortrait's cropper wants InsightFace;
both are restricted for commercial use. This replaces both, is Apache-2.0
throughout, and runs on CPU so it never competes with the GPU the renderers
need.
"""

import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

import facegeom_core as fg

fg.MODEL_PATH = __import__("pathlib").Path(
    os.environ.get("FACE_MODEL_PATH", "/models/face_landmarker.task")
)

app = FastAPI(title="Face geometry")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/detect")
async def detect(file: UploadFile = File(...), include_mask: bool = False):  # noqa: B008
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode that image")

    faces = fg.detect_all(image)

    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "faces": [
            {
                "bbox": list(f.bbox),
                "mouth_box": list(f.mouth_box),
                "mouth_openness": round(f.mouth_openness, 4),
                "yaw": round(f.yaw, 4),
                "landmarks": f.landmarks.astype(float).round(2).tolist(),
                **(
                    {
                        "mask_png_b64": base64.b64encode(
                            cv2.imencode(".png", f.mask)[1].tobytes()
                        ).decode()
                    }
                    if include_mask
                    else {}
                ),
            }
            for f in faces
        ],
    }
