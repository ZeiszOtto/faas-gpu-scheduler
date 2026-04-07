import io
import easyocr
import numpy as np
import cv2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='/app/models')

@app.post("/detect")
async def detect(request: Request):
    image_bytes = await request.body()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"plate_detected": False, "plate_text": "", "confidence": 0.0})

    results = reader.readtext(frame)

    if not results:
        return JSONResponse({"plate_detected": False, "plate_text": "", "confidence": 0.0})

    best = max(results, key=lambda r: r[2])
    confidence = float(best[2])
    text = best[1].strip()

    return JSONResponse({
        "plate_detected": confidence > 0.3,
        "plate_text": text,
        "confidence": round(confidence, 4)
    })