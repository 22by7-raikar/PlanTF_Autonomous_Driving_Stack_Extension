"""Benchmark: mode reranker coverage computation — CPU loop vs GPU kernel.

Compares three implementations of the per-mode on-route coverage calculation:

  1. CPU PyTorch loop   — original baseline (K iterations of torch.cdist)
  2. GPU PyTorch loop   — same logic, tensors moved to CUDA
  3. GPU CUDA kernel    — reranker_cuda_ext.compute_coverage_cuda

Run with:
  conda run -n plantf python benchmarks/bench_reranker.py

The CUDA extension is compiled on first run (~20–30 s); subsequent runs use the
cached .so from ~/.cache/torch_extensions/.

Expected speedup on RTX 3060 Mobile (K=6, T=80, R=30):
  GPU loop    vs CPU loop:  ~2–4 ×   (GPU overhead dominates at small K)
  CUDA kernel vs CPU loop:  ~3–6 ×   (kernel fuses loop + reduction, no python)
  CUDA kernel vs GPU loop:  ~1.5–2 × (eliminates per-k launch overhead)
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.planners.mode_reranker import ROUTE_DIST_THRESHOLD
from src.planners.reranker_cuda import compute_coverage_cuda


# ── Benchmark parameters ─────────────────────────────────────────────────────
K       = 6     # number of trajectory modes
T       = 80    # future steps
R       = 30    # on-route polygon centres (typical real-world value)
WARMUP  = 20
RUNS    = 500


# ── Reference implementations ────────────────────────────────────────────────

def coverage_cpu_loop(wp: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
    """CPU PyTorch loop (original mode_reranker.py baseline)."""
    cov = torch.zeros(K, dtype=torch.float32)
    for k in range(K):
        dists   = torch.cdist(wp[k], rc)
        min_d   = dists.min(dim=-1).values
        cov[k]  = (min_d <= ROUTE_DIST_THRESHOLD).float().mean()
    return cov


def coverage_gpu_loop(wp: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
    """PyTorch loop on GPU (same Python code, tensors on CUDA)."""
    cov = torch.zeros(K, dtype=torch.float32, device=wp.device)
    for k in range(K):
        dists   = torch.cdist(wp[k], rc)
        min_d   = dists.min(dim=-1).values
        cov[k]  = (min_d <= ROUTE_DIST_THRESHOLD).float().mean()
    return cov


# ── Timer helper ──────────────────────────────────────────────────────────────

def _time_fn(fn, *args, label: str) -> float:
    """Run fn(*args) WARMUP+RUNS times; return mean time (ms) over RUNS."""
    for _ in range(WARMUP):
        fn(*args)
    if args and hasattr(args[0], "device") and args[0].is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(*args)
        if args and hasattr(args[0], "device") and args[0].is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    mean_ms = float(np.mean(times))
    p50     = float(np.percentile(times, 50))
    p99     = float(np.percentile(times, 99))
    print(f"  {label:<28s}  mean={mean_ms:6.3f} ms  p50={p50:6.3f}  p99={p99:6.3f}")
    return mean_ms


def _time_cuda_fn(fn, *args, label: str) -> float:
    """Timer for GPU functions — uses CUDA events for sub-ms accuracy."""
    dev = args[0].device
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(RUNS):
        ev_s.record()
        fn(*args)
        ev_e.record()
        torch.cuda.synchronize()
        times.append(ev_s.elapsed_time(ev_e))

    mean_ms = float(np.mean(times))
    p50     = float(np.percentile(times, 50))
    p99     = float(np.percentile(times, 99))
    print(f"  {label:<28s}  mean={mean_ms:6.4f} ms  p50={p50:6.4f}  p99={p99:6.4f}")
    return mean_ms


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*60}")
    print("  PlanTF mode-reranker coverage benchmark")
    print(f"  K={K}  T={T}  R={R}  warmup={WARMUP}  runs={RUNS}")
    print(f"{'='*60}\n")

    torch.manual_seed(0)
    wp_cpu = torch.randn(K, T, 2)
    rc_cpu = torch.randn(R, 2)

    print("─── CPU ───────────────────────────────────────────────")
    t_cpu = _time_fn(coverage_cpu_loop, wp_cpu, rc_cpu,
                     label="CPU PyTorch loop")

    if not torch.cuda.is_available():
        print("\nCUDA not available — skipping GPU benchmarks.\n")
        return

    wp_gpu = wp_cpu.cuda()
    rc_gpu = rc_cpu.cuda()

    print("\n─── GPU ───────────────────────────────────────────────")
    t_gpu_loop = _time_cuda_fn(coverage_gpu_loop, wp_gpu, rc_gpu,
                               label="GPU PyTorch loop")

    # Trigger JIT compilation (reported separately so it doesn't bias timing)
    print("\n  [JIT compile — first call only]")
    t_compile_start = time.perf_counter()
    result = compute_coverage_cuda(wp_gpu, rc_gpu, ROUTE_DIST_THRESHOLD)
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t_compile_start
    if result is None:
        print("  CUDA extension unavailable — skipping kernel benchmark.\n")
        return
    print(f"  JIT compilation took {t_compile:.1f} s\n")

    t_kernel = _time_cuda_fn(
        lambda w, r: compute_coverage_cuda(w, r, ROUTE_DIST_THRESHOLD),
        wp_gpu, rc_gpu,
        label="GPU CUDA kernel",
    )

    # ── Correctness check ────────────────────────────────────────────────────
    ref = coverage_cpu_loop(wp_cpu, rc_cpu)
    cov = compute_coverage_cuda(wp_gpu, rc_gpu, ROUTE_DIST_THRESHOLD)
    assert cov is not None
    max_diff = (cov.cpu() - ref).abs().max().item()
    print(f"\n  Correctness: max |kernel − ref| = {max_diff:.2e}  ", end="")
    print("✓ OK" if max_diff < 1e-4 else "✗ MISMATCH")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Speedups (vs CPU loop, mean latency):")
    print(f"    GPU loop   / CPU loop  : {t_cpu / t_gpu_loop:5.2f} ×")
    print(f"    CUDA kernel / CPU loop : {t_cpu / t_kernel:5.2f} ×")
    print(f"    CUDA kernel / GPU loop : {t_gpu_loop / t_kernel:5.2f} ×")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
