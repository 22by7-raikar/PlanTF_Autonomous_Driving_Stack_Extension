"""Lazy-load the compiled CUDA extension for reranker coverage computation.

The extension is a single .cu file (reranker_cuda_kernel.cu) compiled via
``torch.utils.cpp_extension.load()`` on the first call that needs it.
Subsequent imports reuse the cached shared object from
``~/.cache/torch_extensions/`` (or ``$TORCH_EXTENSIONS_DIR``).

When the extension is unavailable (CUDA not present, compiler absent,
compilation error), all functions degrade to returning ``None`` and the
caller falls back to the PyTorch implementation in mode_reranker.py.

Usage
-----
    from .reranker_cuda import compute_coverage_cuda

    cov = compute_coverage_cuda(traj_xy, route_centers, threshold)
    if cov is None:
        # CUDA extension unavailable — use PyTorch fallback
        ...
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch


# ── Extension loader ─────────────────────────────────────────────────────────

_ext: Optional[object] = None
_tried: bool = False


def _try_load() -> Optional[object]:
    """Attempt to JIT-compile and load the CUDA extension.

    Returns the loaded module on success, or None on failure.
    Failure is non-fatal; a warning is emitted to stderr.
    """
    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load as _cpp_load  # noqa: PLC0415

        _here = os.path.dirname(os.path.abspath(__file__))
        ext = _cpp_load(
            name="reranker_cuda_ext",
            sources=[os.path.join(_here, "reranker_cuda_kernel.cu")],
            verbose=False,
            # -O2        : optimise; --use_fast_math speeds up float ops
            #              (reciprocal, sqrt) which aren't needed here but
            #              the flag also enables fused FP ops globally.
            extra_cuda_cflags=["-O2", "--use_fast_math"],
        )
        return ext
    except Exception as exc:
        warnings.warn(
            f"[reranker_cuda] CUDA extension failed to compile:\n  {exc}\n"
            "Falling back to PyTorch coverage implementation.",
            stacklevel=3,
        )
        return None


def _get_ext() -> Optional[object]:
    """Return the (possibly cached) extension module, loading it if needed."""
    global _ext, _tried
    if not _tried:
        _ext = _try_load()
        _tried = True
    return _ext


# ── Public API ───────────────────────────────────────────────────────────────

def compute_coverage_cuda(
    traj_xy: torch.Tensor,
    route_centers: torch.Tensor,
    threshold: float,
) -> Optional[torch.Tensor]:
    """Compute per-mode on-route coverage using the CUDA kernel.

    Wraps ``reranker_cuda_ext.compute_coverage_cuda``.  Returns ``None``
    rather than raising if the extension is unavailable so that the caller
    can fall back to a pure-PyTorch path without restructuring control flow.

    Parameters
    ----------
    traj_xy : Tensor[K, T, 2], float32, on CUDA device
        xy coordinates of all K × T waypoints (contiguous expected).
    route_centers : Tensor[R, 2], float32, on the same CUDA device
        xy coordinates of on-route polygon centres (contiguous expected).
    threshold : float
        Distance threshold in metres.  A waypoint is "on route" if its
        minimum distance to any route centre ≤ threshold.

    Returns
    -------
    Tensor[K], float32, on the same CUDA device
        Fraction of on-route waypoints per mode.
    None
        When the CUDA extension is unavailable (compilation failed, CUDA
        absent, or tensors not on GPU).
    """
    ext = _get_ext()
    if ext is None:
        return None
    return ext.compute_coverage_cuda(traj_xy, route_centers, float(threshold))
