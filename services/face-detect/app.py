# Configure TensorFlow to allocate GPU memory lazily.
import tensorflow as tf

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[WARNING] Could not set memory growth: {e}", flush=True)

import io
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from retinaface import RetinaFace
from contextlib import asynccontextmanager

# Warmup image size chosen to match the typical YOLO person-crop dimensions.
WARMUP_IMAGE_WIDTH = 240
WARMUP_IMAGE_HEIGHT = 440


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector

    # Trigger model build at startup instead of on first request to make warmup explicit and measurable.
    print("[STARTUP] Building RetinaFace model...", flush=True)
    RetinaFace.build_model()
    detector = RetinaFace

    # Run a dummy inference to trigger cuDNN convolution autotune.
    # Without this, the first real request pays a one-time ~5s tail latency.
    print(f"[STARTUP] Running warmup inference ({WARMUP_IMAGE_WIDTH}x{WARMUP_IMAGE_HEIGHT})...", flush=True)
    # numpy array shape is (height, width, channels)
    dummy_frame = np.zeros((WARMUP_IMAGE_HEIGHT, WARMUP_IMAGE_WIDTH, 3), dtype=np.uint8)
    detector.detect_faces(dummy_frame)
    print("[STARTUP] Warmup complete, ready for requests.", flush=True)

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/detect")
async def detect(request: Request):
    image_bytes = await request.body()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty request body")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    faces = detector.detect_faces(frame)

    if not faces:
        return {"face_detected": False, "confidence": 0.0}

    best_face = max(faces.values(), key=lambda f: f["score"])

    return {
        "face_detected": True,
        "confidence": round(float(best_face["score"]), 4)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)