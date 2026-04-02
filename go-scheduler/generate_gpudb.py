"""
This is an offline tool that extracts GPU specifications from the dbgpu Python
package and generates a gpu_database.json file consumed by the Go scheduler at
startup. Run this script whenever GPU coverage needs to be updated (e.g., new
GPU generation released, new card added to the cluster).

Requirements:
    pip install "dbgpu[fuzz]"

Usage:
    python generate_gpu_db.py                     # default output: gpu_database.json
    python generate_gpu_db.py -o custom_path.json  # custom output path
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
try:
    from dbgpu import GPUDatabase
except ImportError:
    print("Error importing dbgpu package. Install with: pip install 'dbgpu[fuzz]'")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Memory type ordinal encoding
# Captures generational improvements in memory technology.
# ---------------------------------------------------------------------------
MEMORY_TYPE_ORDINAL = {
    "GDDR5":  1,
    "GDDR5X": 2,
    "GDDR6":  3,
    "GDDR6X": 4,
    "GDDR7":  5,
    "HBM2":   5,
    "HBM2e":  6,
    "HBM3":   7,
    "HBM3e":  8,
}

# ---------------------------------------------------------------------------
# Generation filter
# Only focus on NVIDIA GPUs that are likely to appear in a Kubernetes cluster.
# ---------------------------------------------------------------------------
INCLUDED_GENERATIONS = {
    # Desktop consumer
    "GeForce 10",       # Pascal:       GTX 1050 – 1080 Ti
    "GeForce 16",       # Turing:       GTX 1650 – 1660 Ti
    "GeForce 20",       # Turing:       RTX 2060 – 2080 Ti
    "GeForce 30",       # Ampere:       RTX 3050 – 3090 Ti
    "GeForce 40",       # Ada Lovelace: RTX 4060 – 4090
    "GeForce 50",       # Blackwell:    RTX 5050 – 5090

    # Data center / server
    "Tesla Pascal(Pxx)",
    "Tesla Volta(Vxx)",
    "Tesla Turing(Txx)",
    "Server Ampere(Axx)",
    "Server Ada(Lxx)",
    "Server Hopper(Hxx)",
    "Server Blackwell(Bxx)",
}

# ---------------------------------------------------------------------------
# Scoring presets
# Three weight configurations for different workload profiles.
# ---------------------------------------------------------------------------
SCORING_PRESETS = {
    "inference": {
        "description": "Optimized for inference workloads (80% inference / 20% training).",
        "bandwidth_weight":   0.35,
        "vram_weight":        0.15,
        "tensor_weight":      0.20,
        "fp32_weight":        0.20,
        "memory_type_weight": 0.10,
    },
    "balanced": {
        "description": "Balanced for mixed workloads (50% inference / 50% training).",
        "bandwidth_weight":   0.30,
        "vram_weight":        0.25,
        "tensor_weight":      0.20,
        "fp32_weight":        0.15,
        "memory_type_weight": 0.10,
    },
    "training": {
        "description": "Optimized for training workloads (80% training / 20% inference).",
        "bandwidth_weight":   0.25,
        "vram_weight":        0.30,
        "tensor_weight":      0.25,
        "fp32_weight":        0.10,
        "memory_type_weight": 0.10,
    },
}

# Check if a GPU should be included in the scheduler database.
def is_relevant_gpu(gpu) -> bool:
    if gpu.manufacturer != "NVIDIA":
        return False

    if gpu.generation not in INCLUDED_GENERATIONS:
        return False

    if gpu.memory_size_gb is None or gpu.memory_size_gb <= 0:
        return False
    if gpu.memory_bandwidth_gb_s is None or gpu.memory_bandwidth_gb_s <= 0:
        return False
    if gpu.single_float_performance_gflop_s is None:
        return False

    return True

# Extract relevant fields from GPUSpecification object.
def collect_gpu_specs(db: GPUDatabase) -> dict:
    gpu_dict= {}

    for _slug, gpu in sorted(db.specifications.items()):
        if not is_relevant_gpu(gpu):
            continue

        fp32_gflops = gpu.single_float_performance_gflop_s or 0.0
        mem_type = gpu.memory_type or "Unknown"

        gpu_dict[gpu.name] = {
            "architecture":           gpu.architecture or "Unknown",
            "memory_size_gb":         gpu.memory_size_gb,
            "memory_type":            mem_type,
            "memory_type_ordinal":    MEMORY_TYPE_ORDINAL.get(mem_type, 0),
            "memory_bandwidth_gb_s":  gpu.memory_bandwidth_gb_s,
            "tensor_cores":           gpu.tensor_cores,
            "cuda_cores":             gpu.shading_units,
            "fp32_tflops":            round(fp32_gflops / 1000.0, 2),
        }

    return gpu_dict

# Applies a min-max normalization to a list of values, producing [0.0, 1.0] range.
def min_max_normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    diff = hi - lo

    if diff == 0:
        return [0.0] * len(values)
    return [(v - lo) / diff for v in values]

# Compute min-max normalized values for each scoring metric and attach them to every GPU entry.
def normalize_gpu_specs(gpu_dict: dict) -> dict:
    names = list(gpu_dict.keys())

    # Collect raw values in insertion order
    raw_bw    = [gpu_dict[n]["memory_bandwidth_gb_s"] for n in names]
    raw_vram  = [gpu_dict[n]["memory_size_gb"] for n in names]
    raw_tc    = [gpu_dict[n]["tensor_cores"] for n in names]
    raw_fp32  = [gpu_dict[n]["fp32_tflops"] for n in names]
    raw_mtype = [gpu_dict[n]["memory_type_ordinal"] for n in names]

    # log2 transform on VRAM
    log_vram = [math.log2(v) if v > 0 else 0.0 for v in raw_vram]

    # Normalize each metric independently
    norm_bw    = min_max_normalize(raw_bw)
    norm_vram  = min_max_normalize(log_vram)
    norm_tc    = min_max_normalize(raw_tc)
    norm_fp32  = min_max_normalize(raw_fp32)
    norm_mtype = min_max_normalize(raw_mtype)

    # Attach normalized values to each GPU entry
    for i, name in enumerate(names):
        gpu_dict[name]["norm_bandwidth"]   = round(norm_bw[i], 4)
        gpu_dict[name]["norm_vram"]        = round(norm_vram[i], 4)
        gpu_dict[name]["norm_tensor"]      = round(norm_tc[i], 4)
        gpu_dict[name]["norm_fp32"]        = round(norm_fp32[i], 4)
        gpu_dict[name]["norm_memory_type"] = round(norm_mtype[i], 4)

    return gpu_dict

# Assemble the final JSON structure containing scoring presets, and the GPU specification entries.
def build_output_json(gpu_dict: dict) -> dict:
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source":       "dbgpu v2025.12 (https://github.com/painebenjamin/dbgpu)",
            "gpu_count":    len(gpu_dict),
        },
        "scoring_presets": SCORING_PRESETS,
        "gpus": gpu_dict,
    }

# Write the assembled JSON into a file
def write_json(data: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_database(output_path: str) -> None:
    db = GPUDatabase.default()

    gpus = collect_gpu_specs(db)
    if not gpus:
        print("ERROR: No GPUs matched the filter criteria.")
        sys.exit(1)

    gpus = normalize_gpu_specs(gpus)

    output = build_output_json(gpus)

    write_json(output, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU specification database for the GPU-aware scheduler"
    )
    parser.add_argument(
        "-o", "--output",
        default="gpu_database.json",
        help="Output JSON file path (default: gpu_database.json)",
    )
    args = parser.parse_args()
    generate_database(args.output)


if __name__ == "__main__":
    main()