"""
dump_calibration_data.py
------------------------
Generate synthetic calibration batches for TensorRT INT8 PTQ.

Each batch is saved as a NumPy .npy file containing a flat dict
    {input_name: np.ndarray}
with shapes and dtypes matching the ONNX model's input contract.

How TRT INT8 calibration works
------------------------------
TensorRT's IInt8EntropyCalibrator2 calibrates each layer by:
  1. Running the network forward for every calibration batch.
  2. Recording per-layer activation histograms.
  3. Finding the scale factor that minimises KL-divergence between the
     original float32 distribution and the quantized INT8 distribution.

The quality of the calibration (and therefore the fidelity of the INT8
engine) depends on how well the calibration batches represent the real
input distribution.  More batches → better scale estimates.

TRT recommendations: 500–1000 batches is typically sufficient for convergence.

Synthetic vs real-data calibration
-----------------------------------
This script uses SYNTHETIC random inputs with varying agent and polygon
counts.  This is adequate for:
  * latency benchmarking (INT8 kernel selection is the same regardless of
    calibration quality)
  * verifying the calibration pipeline end-to-end

For PRODUCTION deployment replace this script with one that samples real
nuPlan scenarios via the feature extractor.  The output format (flat .npy
dicts of the 23 input tensors) is unchanged; only the generator differs.

The calibration files are large (~2 MB each × 512 batches = ~1 GB) and are
excluded from git via .gitignore (inference/calib_data/).

Usage (from repo root)
----------------------
    # Default: 512 batches → inference/calib_data/
    python inference/deploy/dump_calibration_data.py

    # Smaller dataset for a quick test:
    python inference/deploy/dump_calibration_data.py --n-batches 32

    # Custom output directory:
    python inference/deploy/dump_calibration_data.py --out-dir /tmp/calib

Output
------
Each batch written as inference/calib_data/batch_NNNN.npy
  np.load("batch_0000.npy", allow_pickle=True).item()  →  dict[str, ndarray]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or this directory
# ---------------------------------------------------------------------------
_here      = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)

sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_inference, "common"))

import torch
from run_inference import make_dummy_input

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
_DEFAULT_N_BATCHES    = 512
_DEFAULT_OUT_DIR      = os.path.join(_inference, "calib_data")
_DEFAULT_MIN_AGENTS   = 16   # agents (incl. ego); nuPlan mini typically has 16–33
_DEFAULT_MAX_AGENTS   = 33
_DEFAULT_MIN_POLYGONS = 64   # map polygons; varies per scene
_DEFAULT_MAX_POLYGONS = 200


# ---------------------------------------------------------------------------
# Input flattening
# ---------------------------------------------------------------------------

def _flatten(data: dict) -> dict[str, np.ndarray]:
    """Flatten the nested feature dict to a flat {input_name: ndarray} dict.

    This mirrors _build_trt_feeds in benchmark_tensorrt.py and must stay in
    sync with the ONNX model's input names.  The 23 entries below cover the
    complete planTF ONNX input contract.
    """
    ag = data["agent"]
    mp = data["map"]
    return {
        # ── agent features ──────────────────────────────────────────────────
        "a_position":                ag["position"].numpy(),         # [1,A,21,2] f32
        "a_heading":                 ag["heading"].numpy(),          # [1,A,21]   f32
        "a_velocity":                ag["velocity"].numpy(),         # [1,A,21,2] f32
        "a_shape":                   ag["shape"].numpy(),            # [1,A,21,2] f32
        "a_category":                ag["category"].numpy(),         # [1,A]      i64
        "a_valid_mask":              ag["valid_mask"].numpy(),       # [1,A,21]   bool
        "a_target":                  ag["target"].numpy(),           # [1,A,0,3]  f32
        # ── map features ────────────────────────────────────────────────────
        "m_point_position":          mp["point_position"].numpy(),   # [1,M,3,20,2] f32
        "m_point_vector":            mp["point_vector"].numpy(),     # [1,M,3,20,2] f32
        "m_point_orientation":       mp["point_orientation"].numpy(),# [1,M,3,20]   f32
        "m_point_side":              mp["point_side"].numpy(),       # [1,M,3]      f32
        "m_polygon_center":          mp["polygon_center"].numpy(),   # [1,M,3]      f32
        "m_polygon_position":        mp["polygon_position"].numpy(), # [1,M,2]      f32
        "m_polygon_orientation":     mp["polygon_orientation"].numpy(),# [1,M]      f32
        "m_polygon_type":            mp["polygon_type"].numpy(),     # [1,M]        i64
        "m_polygon_on_route":        mp["polygon_on_route"].numpy(), # [1,M]        bool
        "m_polygon_tl_status":       mp["polygon_tl_status"].numpy(),# [1,M]        i64
        "m_polygon_has_speed_limit": mp["polygon_has_speed_limit"].numpy(),# [1,M]  bool
        "m_polygon_speed_limit":     mp["polygon_speed_limit"].numpy(),# [1,M]      f32
        "m_valid_mask":              mp["valid_mask"].numpy(),       # [1,M,20]     bool
        # ── vehicle state ───────────────────────────────────────────────────
        "current_state":             data["current_state"].numpy(),  # [1,7]   f32
        "origin":                    data["origin"].numpy(),         # [1,2]   f32
        "angle":                     data["angle"].numpy(),          # [1]     f32
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic TRT INT8 calibration batches for planTF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-batches", type=int, default=_DEFAULT_N_BATCHES,
        help=f"Number of calibration batches to generate (default: {_DEFAULT_N_BATCHES}). "
             "TRT recommends ≥500 for good INT8 scale estimation.",
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help=f"Directory for calibration batch files (default: {_DEFAULT_OUT_DIR}). "
             "Excluded from git; may be ~1 GB for 512 batches.",
    )
    parser.add_argument(
        "--min-agents", type=int, default=_DEFAULT_MIN_AGENTS,
        help=f"Minimum agent count per batch (default: {_DEFAULT_MIN_AGENTS})",
    )
    parser.add_argument(
        "--max-agents", type=int, default=_DEFAULT_MAX_AGENTS,
        help=f"Maximum agent count per batch (default: {_DEFAULT_MAX_AGENTS})",
    )
    parser.add_argument(
        "--min-polygons", type=int, default=_DEFAULT_MIN_POLYGONS,
        help=f"Minimum polygon count per batch (default: {_DEFAULT_MIN_POLYGONS})",
    )
    parser.add_argument(
        "--max-polygons", type=int, default=_DEFAULT_MAX_POLYGONS,
        help=f"Maximum polygon count per batch (default: {_DEFAULT_MAX_POLYGONS})",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nGenerating {args.n_batches} synthetic calibration batches → {args.out_dir}")
    print(f"  Agents  : {args.min_agents}–{args.max_agents}")
    print(f"  Polygons: {args.min_polygons}–{args.max_polygons}")
    print()
    print("  NOTE: Synthetic inputs are random tensors, not real nuPlan features.")
    print("  This is adequate for latency benchmarking and pipeline validation.")
    print("  For best INT8 accuracy, replace with real scenario data.")
    print()

    rng = random.Random(0)
    n_written = 0

    for i in range(args.n_batches):
        # Vary the random seed so each batch exercises a different activation
        # regime.  Using the batch index as seed makes generation reproducible.
        torch.manual_seed(i)
        np.random.seed(i)

        A = rng.randint(args.min_agents,   args.max_agents)
        M = rng.randint(args.min_polygons, args.max_polygons)

        data = make_dummy_input(A=A, M=M)
        flat = _flatten(data)

        out_path = os.path.join(args.out_dir, f"batch_{i:04d}.npy")
        np.save(out_path, flat, allow_pickle=True)   # saves the dict as object array
        n_written += 1

        if i == 0 or (i + 1) % 50 == 0:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  [{i + 1:4d}/{args.n_batches}]  A={A:2d}  M={M:3d}  "
                  f"→ {os.path.basename(out_path)}  ({size_kb:.0f} KB)")

    # Compute total directory size
    total_bytes = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
        if f.endswith(".npy")
    )
    total_mb = total_bytes / 1e6

    print(f"\nDone. {n_written} batches written ({total_mb:.0f} MB total)")
    print(f"\nTo build INT8 engine use:")
    print(f"  python inference/deploy/benchmark_tensorrt.py "
          f"--int8 --calib-dir {args.out_dir} --rebuild")
    return 0


if __name__ == "__main__":
    sys.exit(main())
