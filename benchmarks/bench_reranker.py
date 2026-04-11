"""Microbenchmark: loop vs vectorised coverage calculation in mode_reranker.

Motivation
----------
The inner loop in ``mode_reranker.rerank_modes()`` historically called
``torch.cdist`` once per mode (K calls total).  The vectorised refactor
collapses those K calls into one by reshaping the [K, T, 2] waypoints tensor
into [K*T, 2] before the single cdist call.

Each Python→CUDA kernel dispatch carries fixed overhead (launch latency,
Python GIL, stream synchronisation check) that dominates for small tensors
like [T=80, R=~64].  Eliminating K-1 of those dispatches is the whole win.

Usage
-----
CPU benchmark (latencies are meaningful but CUDA savings don't show):
    python benchmarks/bench_reranker.py

CUDA benchmark (shows actual kernel-dispatch savings):
    python benchmarks/bench_reranker.py --device cuda

Custom shapes:
    python benchmarks/bench_reranker.py --device cuda --K 6 --T 80 --R 128

Output
------
A latency table with µs/call, calls/s, and speedup, followed by a
numerical-equivalence check that confirms the two implementations agree.
"""

import argparse
import os
import sys
import time

import torch

# Allow running from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.planners.mode_reranker import ROUTE_DIST_THRESHOLD

# ── default shapes (match a typical nuPlan mini scenario) ────────────────────
_DEFAULT_K = 6     # trajectory modes
_DEFAULT_T = 80    # future timesteps
_DEFAULT_R = 64    # on-route polygon centres (varies; 64 is a mid-scenario count)
_WARMUP    = 200   # iterations discarded before timing
_RUNS      = 10_000


# ── the two implementations being compared ───────────────────────────────────

def _coverage_looped(traj: torch.Tensor, route_centers: torch.Tensor) -> torch.Tensor:
    """Reference: scalar for-loop over K modes (pre-vectorisation code).

    traj         : [K, T, 4]  — all candidate trajectories (x, y, cos_h, sin_h)
    route_centers: [R, 2]     — on-route polygon centres (x, y)
    returns      : [K]        — coverage fraction per mode
    """
    K_ = traj.shape[0]
    coverage = torch.zeros(K_, dtype=torch.float32, device=traj.device)
    for k in range(K_):
        waypoints = traj[k, :, :2]                            # [T, 2]
        dists     = torch.cdist(waypoints, route_centers)     # [T, R]
        min_dists = dists.min(dim=-1).values                  # [T]
        coverage[k] = (min_dists <= ROUTE_DIST_THRESHOLD).float().mean()
    return coverage


def _coverage_vectorized(traj: torch.Tensor, route_centers: torch.Tensor) -> torch.Tensor:
    """Vectorised: single cdist call over all K*T waypoints.

    traj         : [K, T, 4]
    route_centers: [R, 2]
    returns      : [K]
    """
    K_, T_ = traj.shape[0], traj.shape[1]
    waypoints_flat = traj[:, :, :2].reshape(K_ * T_, 2)       # [K*T, 2]
    dists_flat     = torch.cdist(waypoints_flat, route_centers)  # [K*T, R]
    min_dists      = dists_flat.min(dim=-1).values.reshape(K_, T_)  # [K, T]
    return (min_dists <= ROUTE_DIST_THRESHOLD).float().mean(dim=-1)  # [K]


# ── timing helper ─────────────────────────────────────────────────────────────

def _bench(fn, traj, centers, warmup, runs, device):
    """Return mean µs/call over *runs* iterations after *warmup* discard."""
    for _ in range(warmup):
        fn(traj, centers)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn(traj, centers)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / runs * 1_000_000  # µs per call


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark loop vs vectorised coverage in mode_reranker.py"
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Computation device (default: cpu)")
    ap.add_argument("--warmup", type=int, default=_WARMUP,
                    help=f"Warm-up iterations to discard (default: {_WARMUP})")
    ap.add_argument("--runs",   type=int, default=_RUNS,
                    help=f"Timed iterations (default: {_RUNS})")
    ap.add_argument("--K",      type=int, default=_DEFAULT_K,
                    help=f"Number of trajectory modes (default: {_DEFAULT_K})")
    ap.add_argument("--T",      type=int, default=_DEFAULT_T,
                    help=f"Future timesteps per mode (default: {_DEFAULT_T})")
    ap.add_argument("--R",      type=int, default=_DEFAULT_R,
                    help=f"On-route polygon centres (default: {_DEFAULT_R})")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU")
        device = torch.device("cpu")

    torch.manual_seed(0)
    traj    = torch.randn(args.K, args.T, 4, device=device)
    centers = torch.randn(args.R, 2,      device=device)

    print(
        f"\nBenchmark: mode_reranker coverage calculation\n"
        f"  K={args.K}  T={args.T}  R={args.R}  "
        f"device={device.type}  warmup={args.warmup}  runs={args.runs:,}\n"
    )

    us_loop = _bench(_coverage_looped,     traj, centers, args.warmup, args.runs, device)
    us_vec  = _bench(_coverage_vectorized, traj, centers, args.warmup, args.runs, device)
    speedup = us_loop / us_vec if us_vec > 0 else float("inf")

    # ── latency table ─────────────────────────────────────────────────────
    _WL = 40  # label column width
    _WN = 12  # numeric column width
    sep = (
        "+" + "-" * (_WL + 2)
        + "+" + "-" * (_WN + 2)
        + "+" + "-" * (_WN + 2)
        + "+" + "-" * (_WN + 2) + "+"
    )
    hdr = (
        f"| {'Implementation':<{_WL}} "
        f"| {'µs / call':>{_WN}} "
        f"| {'calls / s':>{_WN}} "
        f"| {'speedup':>{_WN}} |"
    )
    print(sep)
    print(hdr)
    print(sep)

    def _row(label, us_val, sp_str):
        cps = f"{1_000_000 / us_val:,.0f}" if us_val > 0 else "n/a"
        print(
            f"| {label:<{_WL}} "
            f"| {us_val:>{_WN}.2f} "
            f"| {cps:>{_WN}} "
            f"| {sp_str:>{_WN}} |"
        )

    _row("loop  (K separate cdist calls)", us_loop, "1.00x")
    _row("vectorised (single cdist call)", us_vec,  f"{speedup:.2f}x")
    print(sep)

    direction = "faster" if speedup >= 1.0 else "slower"
    print(f"\nSpeedup: {speedup:.2f}x  ({direction})")

    # ── numerical equivalence check ───────────────────────────────────────
    # The two implementations perform identical floating-point operations
    # (same order, same precision) so results should be bit-exact.  We use
    # a small tolerance to guard against any future compiler-level reordering.
    with torch.no_grad():
        cov_loop = _coverage_looped(traj, centers)
        cov_vec  = _coverage_vectorized(traj, centers)
    max_diff = (cov_loop - cov_vec).abs().max().item()
    ok = max_diff < 1e-5
    status = "OK" if ok else "MISMATCH — check mode_reranker.py!"
    print(f"\nNumerical equivalence: max |loop - vec| = {max_diff:.2e}  [{status}]")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
