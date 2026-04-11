"""
run_inference.py
----------------
Loads planTF.ckpt and runs one forward pass with dummy tensors that match the
exact shapes observed from the nuPlan simulation pipeline.

No nuPlan dependency required.

Observed shapes (batch=1):
  agent:
    position          [1, A, 21, 2]   float32   (A <= 33, typically  max_agents+1=33)
    heading           [1, A, 21]      float32
    velocity          [1, A, 21, 2]   float32
    shape             [1, A, 21, 2]   float32
    category          [1, A]          float32   (long at model entry)
    valid_mask        [1, A, 21]      bool
    target            [1, A, 0, 3]    float32   (unused at inference)
  map:
    point_position    [1, M, 3, 20, 2] float32
    point_vector      [1, M, 3, 20, 2] float32
    point_orientation [1, M, 3, 20]   float32
    point_side        [1, M, 3]       float32
    polygon_center    [1, M, 3]       float32   (x, y, heading)
    polygon_position  [1, M, 2]       float32
    polygon_orientation [1, M]        float32
    polygon_type      [1, M]          float32   (long at model entry)
    polygon_on_route  [1, M]          bool
    polygon_tl_status [1, M]          float32   (long at model entry)
    polygon_has_speed_limit [1, M]    bool
    polygon_speed_limit [1, M]        float32
    valid_mask        [1, M, 20]      bool
  current_state       [1, 7]          float32
  origin              [1, 2]          float32
  angle               [1]             float32

Outputs:
  trajectory          [1, 6, 80, 4]   float32   (6 modes, 80 steps, x/y/cos/sin)
  probability         [1, 6]          float32
  prediction          [1, A, 80, 2]   float32   (agent future positions)
  output_trajectory   [1, 80, 3]      float32   (best-mode x/y/heading)

Usage:
    python inference/run_inference.py
    python inference/run_inference.py --ckpt path/to/other.ckpt --device cuda
"""

import argparse
import sys
import os
import time
import contextlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_inference = os.path.dirname(_here)
_repo_root = os.path.dirname(_inference)
sys.path.insert(0, _repo_root)

import torch
from src.models.planTF.planning_model import PlanningModel


def load_model(ckpt_path: str, device: torch.device) -> PlanningModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        state_dict = {
            k[len("model."):] if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
    else:
        state_dict = ckpt

    model = PlanningModel()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [ckpt] Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [ckpt] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    model.to(device)
    return model


def make_dummy_input(A: int = 33, M: int = 152, device: torch.device = torch.device("cpu")) -> dict:
    """
    Build a dummy input dict matching the shapes produced by NuplanFeatureBuilder.
    A = number of agents (including ego at index 0), max 33 (max_agents=32 + ego)
    M = number of map polygons, varies per scene
    T = 21 history steps (history_horizon=2s at 10Hz + current = 21)
    P = 20 points per polygon side
    """
    T = 21   # history_steps
    P = 20   # points per polygon side

    data = {
        "agent": {
            "position":   torch.randn(1, A, T, 2, device=device),
            "heading":    torch.randn(1, A, T,    device=device),
            "velocity":   torch.randn(1, A, T, 2, device=device),
            "shape":      torch.randn(1, A, T, 2, device=device),
            "category":   torch.zeros(1, A,        device=device, dtype=torch.long),
            "valid_mask": torch.ones(1, A, T,      device=device, dtype=torch.bool),
            "target":     torch.zeros(1, A, 0, 3,  device=device),        # unused at inference
        },
        "map": {
            "point_position":        torch.randn(1, M, 3, P, 2, device=device),
            "point_vector":          torch.randn(1, M, 3, P, 2, device=device),
            "point_orientation":     torch.randn(1, M, 3, P,    device=device),
            "point_side":            torch.randn(1, M, 3,        device=device),
            "polygon_center":        torch.randn(1, M, 3,        device=device),
            "polygon_position":      torch.randn(1, M, 2,        device=device),
            "polygon_orientation":   torch.randn(1, M,           device=device),
            "polygon_type":          torch.zeros(1, M,           device=device, dtype=torch.long),
            "polygon_on_route":      torch.ones(1, M,            device=device, dtype=torch.bool),
            "polygon_tl_status":     torch.zeros(1, M,           device=device, dtype=torch.long),
            "polygon_has_speed_limit": torch.zeros(1, M,         device=device, dtype=torch.bool),
            "polygon_speed_limit":   torch.zeros(1, M,           device=device),
            "valid_mask":            torch.ones(1, M, P,         device=device, dtype=torch.bool),
        },
        "current_state": torch.randn(1, 7,  device=device),
        "origin":        torch.randn(1, 2,  device=device),
        "angle":         torch.randn(1,     device=device),
    }
    return data


# ---------------------------------------------------------------------------
# ONNX inference helper
# ---------------------------------------------------------------------------

def _build_onnx_feeds(sess, data: dict) -> dict:
    """Build onnxruntime feed dict from only the inputs the ONNX model expects."""
    ag = data["agent"]
    mp = data["map"]
    pool = {
        "a_position":                ag["position"].numpy(),
        "a_heading":                 ag["heading"].numpy(),
        "a_velocity":                ag["velocity"].numpy(),
        "a_shape":                   ag["shape"].numpy(),
        "a_category":                ag["category"].numpy(),
        "a_valid_mask":              ag["valid_mask"].numpy(),
        "a_target":                  ag["target"].numpy(),
        "m_point_position":          mp["point_position"].numpy(),
        "m_point_vector":            mp["point_vector"].numpy(),
        "m_point_orientation":       mp["point_orientation"].numpy(),
        "m_point_side":              mp["point_side"].numpy(),
        "m_polygon_center":          mp["polygon_center"].numpy(),
        "m_polygon_position":        mp["polygon_position"].numpy(),
        "m_polygon_orientation":     mp["polygon_orientation"].numpy(),
        "m_polygon_type":            mp["polygon_type"].numpy(),
        "m_polygon_on_route":        mp["polygon_on_route"].numpy(),
        "m_polygon_tl_status":       mp["polygon_tl_status"].numpy(),
        "m_polygon_has_speed_limit": mp["polygon_has_speed_limit"].numpy(),
        "m_polygon_speed_limit":     mp["polygon_speed_limit"].numpy(),
        "m_valid_mask":              mp["valid_mask"].numpy(),
        "current_state":             data["current_state"].numpy(),
        "origin":                    data["origin"].numpy(),
        "angle":                     data["angle"].numpy(),
    }
    return {inp.name: pool[inp.name] for inp in sess.get_inputs() if inp.name in pool}


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute output comparison metrics between two trajectory arrays."""
    diff = np.abs(a - b)
    return {
        "max_abs":     float(diff.max()),
        "mean_abs":    float(diff.mean()),
        "pos_err_m":   float(np.linalg.norm((a - b)[..., :2], axis=-1).mean()),
        "hdg_err_rad": float(np.abs(a[..., 2] - b[..., 2]).mean()),
    }


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------
# NVTX (NVIDIA Tools Extension) is a C/Python API for inserting named markers
# into a profiling timeline.  When you run the program under `nsys profile`,
# these markers show up as coloured bands in the Nsight Systems UI so you can
# see exactly which part of the code was running at any moment.
#
# torch.cuda.nvtx is available in PyTorch 1.12 but only works when a CUDA
# device is accessible.  We wrap it in a no-op context so CPU-only runs do not
# need a guard at every call site.

@contextlib.contextmanager
def _nvtx_range(name: str, enabled: bool):
    """Context manager that pushes/pops an NVTX range when enabled."""
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def main():
    _default_ckpt = os.path.join(_repo_root, "checkpoints", "planTF.ckpt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=_default_ckpt)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--agents", type=int, default=33,  help="Number of agents (incl. ego)")
    parser.add_argument("--polygons", type=int, default=152, help="Number of map polygons")
    parser.add_argument("--warmup", type=int, default=3,   help="Warmup forward passes")
    parser.add_argument("--runs",   type=int, default=10,  help="Timed forward passes")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help=(
            "Insert NVTX range markers around each pipeline stage. "
            "Only meaningful when running under `nsys profile --trace=nvtx`. "
            "Requires --device cuda."
        ),
    )
    args = parser.parse_args()

    if args.nvtx and args.device != "cuda":
        print("WARNING: --nvtx has no effect without --device cuda")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this machine")

    torch.manual_seed(0)
    device = torch.device(args.device)
    print(f"Device:    {device}")
    print(f"Agents:    {args.agents}")
    print(f"Polygons:  {args.polygons}")

    print(f"\nLoading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device)
    print("Model loaded OK")

    data = make_dummy_input(A=args.agents, M=args.polygons, device=device)

    print("\n=== Input shapes ===")
    for k, v in data.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:30s} {str(vv.shape):30s} {vv.dtype}")
        else:
            print(f"  {k:32s} {str(v.shape):30s} {v.dtype}")

    use_nvtx = args.nvtx and device.type == "cuda"

    with torch.no_grad():
        # ── Warmup (not profiled — let the GPU reach steady state first) ──────
        for _ in range(args.warmup):
            _ = model(data)

        # ── Timed runs with NVTX ranges ──────────────────────────────────────
        # Each run is wrapped in a top-level "plantf_forward" range so the
        # global timeline shows individual iterations.  Inside that we have
        # two sub-ranges:
        #
        #  h2d_transfer  — moving the input dict from CPU to GPU.  In this
        #                  headless script the dummy tensors are already on the
        #                  right device, so this range captures close-to-zero
        #                  work.  In the live planner the equivalent range
        #                  covers the real host-to-device copy.
        #
        #  model_forward — the actual GPU compute for PlanningModel.forward().
        #                  The synchronize() calls before/after ensure the GPU
        #                  has truly finished before we stop timing.
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            with _nvtx_range("plantf_forward", use_nvtx):
                with _nvtx_range("h2d_transfer", use_nvtx):
                    # Dummy data is already on-device; this range is a
                    # placeholder so the profiling structure matches the live
                    # planner where a real H2D copy happens here.
                    pass
                with _nvtx_range("model_forward", use_nvtx):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    out = model(data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / args.runs * 1000

    print("\n=== Output shapes ===")
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k:30s} {str(v.shape):30s} {v.dtype}")

    print(f"\n=== Latency ({args.runs} runs, {args.warmup} warmup) ===")
    print(f"  Mean forward pass: {elapsed:.2f} ms")
    print(f"\noutput_trajectory sample (first 5 steps):")
    print(out["output_trajectory"][0, :5].detach().cpu().numpy())


if __name__ == "__main__":
    main()
