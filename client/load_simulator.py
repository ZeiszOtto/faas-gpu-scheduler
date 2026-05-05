"""
Reads pre-generated YOLO crops from a directory, parses class and frame index from the filename
(e.g. "46_person_0.78.jpg"), and dispatches them to the appropriate Knative service over HTTP at
a controlled rate using a thread pool.

Each request's outcome is logged as a single CSV row including request and
response timestamps, latency, HTTP status, and any error.

Usage:
    python load_simulator.py --total 300 --rate 5
    python load_simulator.py --total 100 --rate 0                    # burst mode
    python load_simulator.py --total 1000 --rate 10 --workers 100
    python load_simulator.py --total 50 --rate 2 --class person      # face-detect only
    python load_simulator.py --total 50 --rate 2 --class vehicle     # plate-detect only
"""
import argparse
import csv
import re
import sys
import threading
import time
import random
import yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

CONFIG = "config.yaml"

CSV_FIELDS = [
    "submitted_at", "completed_at", "frame", "class", "confidence",
    "target", "status_code", "latency_ms", "response", "error",
]

# Filename pattern: <frame>_<class>_<confidence>.jpg, e.g. "46_person_0.78.jpg"
FILENAME_RE = re.compile(r"^(\d+)_(\w+)_([\d.]+)\.jpg$")

# Allowed values for --class filter
CLASS_FILTER_CHOICES = ["all", "person", "vehicle"]

# Argument parser
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load simulator for the FaaS GPU scheduler.")
    parser.add_argument("--total", type=int, required=True,
                        help="Total number of requests to send.")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Target requests per second (0 = burst mode, no pacing).")
    parser.add_argument("--workers", type=int, default=50,
                        help="Thread pool size for concurrent dispatches (default: 50).")
    parser.add_argument("--class", dest="class_filter", type=str, default="all",
                        choices=CLASS_FILTER_CHOICES,
                        help="Only dispatch crops of this class (default: all).")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Random seed for shuffling crop order. If None, crops are in frame order. "
                             "Use the same seed across measurement cells for consistent workload composition.")
    parser.add_argument("--config", type=str, default=CONFIG,
                        help=f"Path to client config YAML (default: {CONFIG}).")
    return parser.parse_args()

# Loads the configuration file
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Scan the directory for YOLO crop JPEGs and return a sorted list of
# {path, frame, class, confidence, image_bytes} dicts. Image bytes are loaded
# into memory to remove disk I/O from the request critical path.
# If shuffle_seed is provided, the list is randomly shuffled with that seed for
# reproducible mixing of person and vehicle crops; otherwise it is sorted by frame index.
def discover_crops(directory: str, class_filter: str = "all",
                   shuffle_seed: Optional[int] = None) -> list[dict]:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Crop directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    crops = []
    for p in dir_path.glob("*.jpg"):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        frame, cls, conf = m.groups()
        if class_filter != "all" and cls != class_filter:
            continue
        with open(p, "rb") as f:
            crops.append({
                "path": str(p),
                "frame": int(frame),
                "class": cls,
                "confidence": float(conf),
                "image_bytes": f.read(),
            })

    if not crops:
        filter_msg = f" matching class={class_filter}" if class_filter != "all" else ""
        print(f"No valid crops found in {directory}{filter_msg}", file=sys.stderr)
        sys.exit(1)

    if shuffle_seed is not None:
        # Sort first to guarantee a deterministic starting order regardless of
        # filesystem listing variability, then shuffle with the seed.
        crops.sort(key=lambda c: c["frame"])
        random.Random(shuffle_seed).shuffle(crops)
    else:
        crops.sort(key=lambda c: c["frame"])

    return crops

# Maps a YOLO class name to the corresponding Knative service URL.
def build_router(cfg: dict) -> dict:
    return {
        "person":  cfg["services"]["face_detect"],
        "vehicle": cfg["services"]["plate_detect"],
    }

# Send a single crop to the target service and append a CSV row with the outcome.
def dispatch_request(crop: dict, target_url: str, session: requests.Session,
                     csv_writer, csv_lock: threading.Lock):
    submitted_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    response_text = ""
    error_text = ""

    try:
        t0 = time.perf_counter()
        resp = session.post(
            target_url,
            data=crop["image_bytes"],
            headers={"Content-Type": "image/jpeg"},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        status_code = resp.status_code
        response_text = resp.text[:200].strip()
    except requests.exceptions.RequestException as e:
        error_text = str(e)

    completed_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    row = {
        "submitted_at": submitted_at,
        "completed_at": completed_at,
        "frame": crop["frame"],
        "class": crop["class"],
        "confidence": f"{crop['confidence']:.2f}",
        "target": target_url,
        "status_code": status_code if status_code is not None else "",
        "latency_ms": f"{latency_ms:.1f}" if latency_ms is not None else "",
        "response": response_text,
        "error": error_text,
    }

    with csv_lock:
        csv_writer.writerow(row)

def run(args: argparse.Namespace):
    cfg = load_config(args.config)
    save_dir = cfg["output"]["save_dir"]
    crops = discover_crops(save_dir, args.class_filter, args.shuffle_seed)
    router = build_router(cfg)

    # Per-run timestamped CSV so multiple runs do not overwrite each other
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_csv = Path(save_dir) / f"load_results_{run_ts}.csv"

    csv_file = open(output_csv, "w", newline="", buffering=1, encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    csv_lock = threading.Lock()

    # Shared HTTP session for connection pooling and keep-alive
    session = requests.Session()

    print(f"Loaded {len(crops)} crops from {save_dir}/ "
          f"(class filter: {args.class_filter}, shuffle seed: {args.shuffle_seed})")
    print(f"Sending {args.total} requests "
          f"(rate={args.rate if args.rate > 0 else 'burst'} req/s, workers={args.workers})")
    print(f"Output CSV: {output_csv}")

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    next_deadline = time.perf_counter()
    submitted = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in range(args.total):
                crop = crops[i % len(crops)]
                target_url = router.get(crop["class"])
                if target_url is None:
                    print(f"Warning: no route for class={crop['class']}, skipping", file=sys.stderr)
                    continue

                pool.submit(dispatch_request, crop, target_url, session, csv_writer, csv_lock)
                submitted += 1

                if interval > 0:
                    next_deadline += interval
                    sleep_for = next_deadline - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            print(f"Submitted {submitted} requests, waiting for completions...")
    finally:
        csv_file.close()
        session.close()
        print(f"Done. Results: {output_csv}")

if __name__ == "__main__":
    run(parse_args())