import cv2
import requests
import yaml
import csv
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO

# Constants
# ---------------------------------------------------------------------------
CONFIG = "config.yaml"
RESULTS_CSV_NAME = "results.csv"
PERSON_CLASS = 0
CAR_CLASSES  = {2, 3, 5, 7}  # car, motorcycle, bus, truck
CSV_FIELDS = [
    "timestamp", "frame", "class", "confidence", "mode", "target",
    "status_code", "latency_ms", "response", "error",
]
# ---------------------------------------------------------------------------


# Loading of configuration file (Uses config.yaml unless specified otherwise)
def load_config(path: str = CONFIG) -> dict:
    with open(path, "r") as conf:
        return yaml.safe_load(conf)


# Setup of logger
def setup_csv_writer(output_path: str):
    f = open(output_path, "w", newline="", buffering=1, encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return f, writer


# Wipes and recreates the output directory so each run starts clean.
def prepare_output_dir(save_dir: str):
    output_path = Path(save_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


# Parses crop normalization settings from the config file
# Returns (person_target, vehicle_target, letterbox) where each *_target
# is a (width, height) tuple, or None if normalization is disabled.
def parse_crop_processing(cfg: dict):
    norm_cfg = cfg.get("crop_processing", {})
    if not norm_cfg.get("enabled", False):
        return None, None, True

    letterbox = norm_cfg.get("letterbox", True)

    person_size = norm_cfg.get("person", {}).get("target_size")
    person_target = tuple(person_size) if person_size else None

    vehicle_size = norm_cfg.get("vehicle", {}).get("target_size")
    vehicle_target = tuple(vehicle_size) if vehicle_size else None

    return person_target, vehicle_target, letterbox


# Opens the video file and returns a VideoCapture. Exits on failure.
def open_video(source: str) -> cv2.VideoCapture:
    if not Path(source).exists():
        print(f"[ERROR] Video file not found: {source}", file=sys.stderr)
        sys.exit(1)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[ERROR] Cannot open video: {source}", file=sys.stderr)
        sys.exit(1)

    return capture


# Resize crop to target_size, preserving aspect ratio with letterbox padding.
# If letterbox is False, the crop is stretched to target_size, distorting
# aspect ratio. With letterbox=True (default), the crop is scaled to fit
# inside target_size and the remainder is padded with black pixels.
def normalize_crop_size(crop, target_size, letterbox: bool = True):
    if not letterbox:
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    target_w, target_h = target_size
    h, w = crop.shape[:2]

    # Scale to fit inside target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    # Pad with black border to reach target_size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


# Extract bounding box region from frame, optionally normalize, return JPEG bytes.
def extract_crop(frame, box, target_size=None, letterbox: bool = True) -> bytes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]

    if target_size is not None:
        crop = normalize_crop_size(crop, target_size, letterbox=letterbox)

    _, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return encoded.tobytes()


# Sends cropped image to target service or saves to disk and logs result row to CSV.
def dispatch(url: str,          image_bytes: bytes,     csv_writer,
             cls_name: str,     confidence: float,      frame_idx: int,
             mode: str,         save_dir: str):

    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    if mode == "save":
        filename = f"{save_dir}/{frame_idx}_{cls_name}_{confidence:.2f}.jpg"
        with open(filename, "wb") as f:
            f.write(image_bytes)
        csv_writer.writerow({
            "timestamp": ts,
            "frame": frame_idx,
            "class": cls_name,
            "confidence": f"{confidence:.2f}",
            "mode": "save",
            "target": filename,
        })
        return

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            url,
            data=image_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        csv_writer.writerow({
            "timestamp": ts,
            "frame": frame_idx,
            "class": cls_name,
            "confidence": f"{confidence:.2f}",
            "mode": "http",
            "target": url,
            "status_code": resp.status_code,
            "latency_ms": f"{latency_ms:.1f}",
            "response": resp.text[:200].strip(),
        })
    except requests.exceptions.RequestException as e:
        csv_writer.writerow({
            "timestamp": ts,
            "frame": frame_idx,
            "class": cls_name,
            "confidence": f"{confidence:.2f}",
            "mode": "http",
            "target": url,
            "error": str(e),
        })


# Main pipeline: reads video, runs YOLO and dispatches crops to services or disk.
def run(config_path: str = "config.yaml"):
    # Configuration and output paths
    cfg = load_config(config_path)
    mode = cfg["mode"]
    save_dir = cfg["output"]["save_dir"]
    csv_path = f"{save_dir}/{RESULTS_CSV_NAME}"

    # Service URLs
    face_url = cfg["services"]["face_detect"]
    plate_url = cfg["services"]["plate_detect"]

    # YOLO configuration
    model = YOLO(cfg["yolo"]["model"])
    conf_thresh = cfg["yolo"]["confidence_threshold"]
    device = cfg["yolo"]["device"]
    frame_skip = cfg["video"]["frame_skip"]

    # Filtering and tracking parameters
    min_person_height = cfg["filtering"]["min_person_height"]
    min_vehicle_width = cfg["filtering"]["min_vehicle_width"]
    cooldown_frames = cfg["tracking"]["cooldown_frames"]

    # Crop normalization
    person_target, vehicle_target, letterbox = parse_crop_processing(cfg)
    norm_status = "enabled" if (person_target or vehicle_target) else "disabled"

    # Write startup info to console
    print(f"[INFO] Pipeline starting — source={cfg['video']['source']} "
          f"model={cfg['yolo']['model']} device={device}")
    print(f"[INFO] Mode={mode}, results CSV={csv_path}")
    print(f"[INFO] Crop normalization {norm_status} "
          f"(person={person_target}, vehicle={vehicle_target}, letterbox={letterbox})")

    # Prepare output: wipe previous run, create fresh dir, open CSV and video
    prepare_output_dir(save_dir)
    csv_file, csv_writer = setup_csv_writer(csv_path)
    capture = open_video(cfg["video"]["source"])

    # Processing loop
    last_sent = {}
    frame_idx = 0

    try:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            frame_idx += 1

            # YOLO inference runs on every frame so that track IDs stay consistent
            # (persist=True). Frame skipping is applied AFTER inference, only
            # to throttle dispatch frequency.
            results = model.track(frame, conf=conf_thresh, device=device,
                                  verbose=False, persist=True, iou=0.3)

            if results[0].boxes.id is None:
                continue

            if frame_idx % frame_skip != 0:
                continue

            for box in results[0].boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1

                # Size filtering: skip too-small detections
                if cls_id == PERSON_CLASS and height < min_person_height:
                    continue
                if cls_id in CAR_CLASSES and width < min_vehicle_width:
                    continue

                # Per-track cooldown: avoid spamming the same object every frame
                if track_id in last_sent and frame_idx - last_sent[track_id] < cooldown_frames:
                    continue
                last_sent[track_id] = frame_idx

                if cls_id == PERSON_CLASS:
                    crop = extract_crop(frame, box, target_size=person_target, letterbox=letterbox)
                    dispatch(face_url, crop, csv_writer, "person", confidence, frame_idx, mode, save_dir)
                elif cls_id in CAR_CLASSES:
                    crop = extract_crop(frame, box, target_size=vehicle_target, letterbox=letterbox)
                    dispatch(plate_url, crop, csv_writer, "vehicle", confidence, frame_idx, mode, save_dir)

    finally:
        capture.release()
        csv_file.close()
        print(f"[INFO] Pipeline finished — processed {frame_idx} frames, CSV: {csv_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)