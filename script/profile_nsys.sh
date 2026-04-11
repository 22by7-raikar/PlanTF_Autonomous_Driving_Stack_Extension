#!/usr/bin/env bash
# =============================================================================
# profile_nsys.sh
# =============================================================================
# Produce a Nsight Systems profile of the planTF GPU inference pipeline.
#
# What this does
# --------------
# Runs `nsys profile` around `run_inference.py --device cuda --nvtx`.
# The --nvtx flag activates the NVTX range markers that were added in
# feature/nvtx-nsys, so the resulting timeline shows four named bands:
#
#   plantf_forward  (top-level per-iteration wrapper)
#   ├─ h2d_transfer  (host-to-device data movement)
#   └─ model_forward (GPU inference; synchronize() gates both edges)
#
# The rerank_modes range is present in the live planner (imitation_planner.py)
# but not in the headless run_inference.py driver, because the headless script
# does not call the reranker.  To profile the full pipeline including the
# reranker, wire imitation_planner.py into a nuPlan simulation run and set
# use_nvtx=True in the planner config.
#
# Output
# ------
# outputs/profiles/plantf_YYYYMMDD_HHMMSS.nsys-rep   — open in Nsight Systems GUI
# outputs/profiles/plantf_YYYYMMDD_HHMMSS.sqlite     — auto-generated database
# Console also prints a --stats=true text summary.
#
# Prerequisites
# -------------
#   nsys  must be on PATH  (part of the Nsight Systems installation)
#   The plantf conda env must be available
#   A CUDA device must be accessible (check: nvidia-smi)
#
#   If nsys is not found, this script prints a setup guide and exits.
#
# Usage
# -----
#   bash script/profile_nsys.sh
#   bash script/profile_nsys.sh --runs 200 --warmup 20   # extra args forwarded
#
# The extra args are forwarded directly to run_inference.py, so any flag
# accepted by that script (--ckpt, --agents, --polygons, --warmup, --runs)
# can be passed here.
#
# Merge / branch context
# ----------------------
# This script belongs to feature/nvtx-nsys → inference-extensions.
# Do NOT run this script on main; profiles are git-ignored artefacts.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (works whether called from repo root or from script/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Check for nsys
# ---------------------------------------------------------------------------
if ! command -v nsys &>/dev/null; then
    echo ""
    echo "ERROR: nsys not found on PATH."
    echo ""
    echo "Nsight Systems is part of the CUDA toolkit or can be downloaded from:"
    echo "  https://developer.nvidia.com/nsight-systems"
    echo ""
    echo "Typical install locations:"
    echo "  /opt/nvidia/nsight-systems/*/bin/nsys"
    echo "  /usr/local/cuda/bin/nsys"
    echo ""
    echo "Once installed, add its bin/ directory to PATH, then re-run this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR="$REPO_ROOT/outputs/profiles"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_PREFIX="$OUT_DIR/plantf_${STAMP}"

# ---------------------------------------------------------------------------
# Forward any extra CLI args to run_inference.py
# Defaults: 10 warmup iterations (let GPU reach steady state) + 100 timed runs
# (enough to see stable NVTX bands without spending too long profiling).
# Override with e.g.:  bash profile_nsys.sh --runs 200 --warmup 20
# ---------------------------------------------------------------------------
FORWARD_ARGS="--warmup 10 --runs 100 $*"

echo ""
echo "========================================================================"
echo "  planTF Nsight Systems profile"
echo "========================================================================"
echo "  Output prefix : $OUT_PREFIX"
echo "  Forward args  : $FORWARD_ARGS"
echo ""
echo "  Trace spans:"
echo "    CUDA kernels     (--trace=cuda)"
echo "    NVTX ranges      (--trace=nvtx)  ← named bands from run_inference.py"
echo "    OS runtime       (--trace=osrt)  ← pthread, malloc, etc."
echo "    CUDA memory      (--cuda-memory-usage=true)"
echo ""
echo "  This will run $FORWARD_ARGS on a CUDA device."
echo "  Press Ctrl+C to cancel, or wait for profiling to begin ..."
echo ""

# ---------------------------------------------------------------------------
# nsys profile command
# ---------------------------------------------------------------------------
# Flag rationale:
#   --trace=cuda,nvtx,osrt
#       cuda  : captures every kernel launch, memcpy, synchronize
#       nvtx  : captures the range_push/pop markers we inserted
#       osrt  : captures OS-level thread activity; useful to see CPU/GPU overlap
#
#   --cuda-memory-usage=true
#       Shows cudaMalloc / cudaFree events in the timeline.
#       Useful for spotting unexpected allocations inside the inference loop.
#
#   --gpu-metrics-device=all
#       Collects GPU utilization, SM activity, and memory bandwidth counters
#       as time-series overlays in the timeline (requires Nsight >= 2022.x).
#       If incompatible with your nsys version, remove this flag.
#
#   --stats=true
#       Writes a human-readable text report to stdout alongside the .nsys-rep.
#       The report shows per-kernel duration totals, which is useful for a
#       quick CLI read without opening the GUI.
#
#   --output
#       Writes two files: .nsys-rep (GUI) and .sqlite (database).
#       Both are git-ignored via the pattern outputs/profiles/*.
# ---------------------------------------------------------------------------
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --stats=true \
    --output="$OUT_PREFIX" \
    conda run -n plantf \
        python "$REPO_ROOT/inference/common/run_inference.py" \
            --device cuda \
            --nvtx \
            $FORWARD_ARGS

echo ""
echo "========================================================================"
echo "  Profile complete."
echo "  nsys-rep : ${OUT_PREFIX}.nsys-rep"
echo "  sqlite   : ${OUT_PREFIX}.sqlite"
echo ""
echo "  Open with: nsys-ui ${OUT_PREFIX}.nsys-rep"
echo ""
echo "  What to look for in the timeline:"
echo "    1. Duration of 'feature_build' vs 'model_forward'"
echo "       (shows CPU preprocessing overhead vs GPU inference time)"
echo "    2. Gap between 'h2d_transfer' end and 'model_forward' start"
echo "       (synchronisation bubble from pageable host memory)"
echo "    3. Duration of 'rerank_modes' in imitation_planner.py runs"
echo "       (not visible in headless run; run via nuPlan sim with use_nvtx=True)"
echo "    4. Any GPU idle time between iterations"
echo "       (indicates synchronisation points that could be overlapped)"
echo "========================================================================"
