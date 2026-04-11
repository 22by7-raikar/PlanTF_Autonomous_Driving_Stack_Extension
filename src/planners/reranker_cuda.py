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


def _find_cuda_includes() -> list[str]:
    """Return a list of include directories needed to compile the CUDA extension.

    Resolves, in order:
      1. ``site-packages/nvidia/cuda_runtime/include``  — ``cuda_runtime.h``
      2. ``$CUDA_HOME/include`` / ``$CUDA_PATH/include``
      3. Same candidates for ``thrust/complex.h`` (Thrust is part of the CUDA
         toolkit but absent from the pip-installed cuda_runtime package).
    """
    import sysconfig  # noqa: PLC0415

    includes: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if path not in seen and os.path.isdir(path):
            includes.append(path)
            seen.add(path)

    site_pkgs = sysconfig.get_paths()["purelib"]

    # ── Candidate roots that may contain CUDA headers ────────────────────────
    candidates: list[str] = []

    # pip-installed nvidia packages (cuda_runtime, cuda_nvcc)
    for pkg in ("cuda_runtime", "cuda_nvcc"):
        candidates.append(os.path.join(site_pkgs, "nvidia", pkg, "include"))

    # CUDA_HOME / CUDA_PATH env vars
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_var, "")
        if cuda_root:
            candidates.append(os.path.join(cuda_root, "include"))

    # ── Add dirs that satisfy at least one of our known required headers ──────
    required = {
        "cuda_runtime.h": False,
        os.path.join("thrust", "complex.h"): False,
    }
    for root in candidates:
        for header in list(required.keys()):
            if not required[header] and os.path.isfile(os.path.join(root, header)):
                _add(root)
                required[header] = True

    # ── Fallback: check every readable CUDA include we can find on disk ───────
    # This covers site-local CUDA installs (e.g. MATLAB's toolkit).
    if not all(required.values()):
        _fallback_roots = [
            # MATLAB ships an almost-complete CUDA toolkit
            "/home/apr/apps/MATLAB/R2025b/sys/cuda/glnxa64/cuda/include",
            "/usr/local/cuda/include",
            "/usr/cuda/include",
        ]
        for root in _fallback_roots:
            for header in list(required.keys()):
                if not required[header] and os.path.isfile(
                    os.path.join(root, header)
                ):
                    _add(root)
                    required[header] = True

    return includes


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

        # Collect CUDA header directories that may live outside $CUDA_HOME
        # (e.g. pip-installed cuda_runtime, MATLAB toolkit for Thrust).
        extra_includes = _find_cuda_includes()

        # Collect CUDA library directories needed for linking (-lcudart).
        extra_ldflags: list[str] = []
        import sysconfig  # noqa: PLC0415

        site_pkgs = sysconfig.get_paths()["purelib"]
        cudart_lib = os.path.join(site_pkgs, "nvidia", "cuda_runtime", "lib")
        if os.path.isfile(os.path.join(cudart_lib, "libcudart.so")):
            extra_ldflags.append(f"-L{cudart_lib}")

        ext = _cpp_load(
            name="reranker_cuda_ext",
            sources=[os.path.join(_here, "reranker_cuda_kernel.cu")],
            verbose=False,
            # -O2        : optimise; --use_fast_math speeds up float ops
            #              (reciprocal, sqrt) which aren't needed here but
            #              the flag also enables fused FP ops globally.
            extra_cuda_cflags=["-O2", "--use_fast_math"],
            extra_include_paths=extra_includes,
            extra_ldflags=extra_ldflags,
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
