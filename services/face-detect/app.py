import io
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Request
from retinaface import RetinaFace
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = RetinaFace
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