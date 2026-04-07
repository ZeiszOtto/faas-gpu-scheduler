import cv2
import requests
import yaml
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO

CONFIG = "config.yaml"
PERSON_CLASS = 0
CAR_CLASSES  = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Loading of configuration file (Uses config.yaml unless specified otherwise)
def load_config(path: str = CONFIG) -> dict:
    with open(path, "r") as conf:
        return yaml.safe_load(conf)


# Setup of logger
def setup_logger(output: str) -> logging.Logger:
    logger = logging.getLogger("client")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)

    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# Function to open video file and return a VideoCapture object
def open_video(source: str, logger: logging.Logger) -> cv2.VideoCapture:
    if not Path(source).exists():
        logger.error(f"Video file not found: {source}")
        sys.exit(1)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        logger.error(f"Cannot open video: {source}")
        sys.exit(1)

    return capture


# Function to extract the crop from the frame and return it as bytes
def extract_crop(frame, box) -> bytes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    _, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes()


# Function to send the cropped image to the target
def dispatch(url: str,       image_bytes: bytes,  logger: logging.Logger,
             cls_name: str,  confidence: float,   frame_idx: int,
             mode: str,      save_dir: str = "crops"):

    if mode == "save":
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{save_dir}/{frame_idx}_{cls_name}_{confidence:.2f}.jpg"
        with open(filename, "wb") as f:
            f.write(image_bytes)
        logger.info(
            f"frame={frame_idx} class={cls_name} conf={confidence:.2f} "
            f"saved={filename}"
        )
        return

    send_ts = datetime.now(timezone.utc).isoformat()
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            url,
            data=image_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"frame={frame_idx} class={cls_name} conf={confidence:.2f} "
            f"sent_at={send_ts} target={url} "
            f"status={resp.status_code} latency_ms={latency_ms:.1f} "
            f"response={resp.text[:200].strip()!r}"
        )
    except requests.exceptions.RequestException as e:
        logger.warning(
            f"frame={frame_idx} class={cls_name} conf={confidence:.2f} "
            f"sent_at={send_ts} target={url} error={e}"
        )


# ----- YOLO detection logic
def run(config_path: str = "config.yaml"):
    # ----- Config and logger initialization
    cfg      = load_config(config_path)
    mode     = cfg["mode"]
    run_ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_dir = f"{cfg['output']['save_dir']}_{run_ts}"
    log_file = f"results_{run_ts}.log"
    logger   = setup_logger(log_file)

    face_url  = cfg["services"]["face_detect"]
    plate_url = cfg["services"]["plate_detect"]

    model       = YOLO(cfg["yolo"]["model"])
    conf_thresh = cfg["yolo"]["confidence_threshold"]
    device      = cfg["yolo"]["device"]
    frame_skip  = cfg["video"]["frame_skip"]

    min_person_height = cfg["filtering"]["min_person_height"]
    min_vehicle_width = cfg["filtering"]["min_vehicle_width"]

    # ----- Capture and processing loop
    capture   = open_video(cfg["video"]["source"], logger)
    last_sent = {}
    logger.info(f"Starting pipeline — source={cfg['video']['source']} model={cfg['yolo']['model']} device={device}")

    frame_idx = 0
    try:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame_idx += 1

            results = model.track(frame, conf=conf_thresh, device=device, verbose=False, persist=True, iou=0.3)

            if results[0].boxes.id is None:
                continue
            
            if frame_idx % frame_skip != 0:
                continue

            for box in results[0].boxes:
                track_id   = int(box.id[0])
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width  = x2 - x1
                height = y2 - y1

                if cls_id == PERSON_CLASS and height < min_person_height:
                    continue

                if cls_id in CAR_CLASSES and width < min_vehicle_width:
                    continue

                if track_id in last_sent and frame_idx - last_sent[track_id] < cfg["tracking"]["cooldown_frames"]:
                    continue
                last_sent[track_id] = frame_idx

                if cls_id == PERSON_CLASS:
                    crop = extract_crop(frame, box)
                    dispatch(face_url, crop, logger, "person", confidence, frame_idx, mode, save_dir)

                elif cls_id in CAR_CLASSES:
                    crop = extract_crop(frame, box)
                    dispatch(plate_url, crop, logger, "vehicle", confidence, frame_idx, mode, save_dir)

    finally:
        capture.release()
        logger.info(f"Pipeline finished — processed {frame_idx} frames")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)