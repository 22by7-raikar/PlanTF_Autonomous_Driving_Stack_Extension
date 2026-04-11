# Branching Guide

## Branch Overview

```
main  ──────────────────────────────────────────────── all feature branches merged
  (expanded-eval   merged March 2026, commit 46792b4)
  (cpp-runtime     merged March 2026, commit 93b367b)
  (onnx-deployment legacy, content in inference/)
  (faithful-natten legacy, content in inference/faithful/)

inference-extensions  ─── stable base for the five 2026-04 feature branches below
  feature/nvtx-nsys               (commit b749d40)
  feature/reranker-vectorization  (commit c593f62)
  feature/int8-ptq                (commit d521733)
  feature/cpp-runtime-optimizations (commit cd80b75)
  feature/cuda-reranker           (commit 3584513)
```

---

## What Belongs Where

### `main` — Stable baseline
- **Always runnable**: `bash script/initial_sanity_check.sh` and `bash script/mini_benchmark.sh` must pass.
- **Only merge via `--no-ff`** from a feature branch after review.
- **No in-progress work** — nothing half-implemented lives here.
- **Contains**: the 10-scenario `mini_benchmark.yaml` and its frozen `baseline.md` results.

### `expanded-eval` — Broader scenario benchmark
**Purpose**: coverage-driven evaluation beyond the 10-scenario dev benchmark.

Files that belong here:
- `config/scenario_filter/mini_eval_v2.yaml`
- `script/mini_eval_v2.sh`
- `script/summarize_eval.py`
- `docs/mini_scenario_inventory.md` / `outputs/mini_scenario_inventory.csv`
- `docs/EVAL_PLAN.md`
- Results under `outputs/eval_v2/`

**Do NOT mix in**:
- C++ code of any kind
- Changes to `inference/common/patches.py` or any planner logic
- Deployment artifacts (`.trt`, `.onnx`)

### `cpp-runtime` — Native C++ inference
**Purpose**: ONNX Runtime C++ first, TensorRT C++ second.

Files that belong here:
- `cpp/onnx_runtime_infer.cpp`
- `cpp/tensorrt_infer.cpp`
- `cpp/tensorrt_utils.h` / `.cpp`
- `cpp/CMakeLists.txt`
- `cpp/README.md`
- `cpp/dummy_input_writer.py` (helper to write test inputs)

**Do NOT mix in**:
- Any change to planner behaviour or `imitation_planner.py`
- Scenario filter changes
- Faithful/patch research
- Result claims from expanded eval

**Uses**: `inference/planTF.onnx` (patched export) — must never be confused
with the original unpatched model.

### `onnx-deployment` — Python ONNX/TRT benchmarking (legacy)
All previous Python TensorRT work is here. Keep stable. New deployment work
may start here or in `cpp-runtime`.

### `faithful-natten` — NATTEN fidelity research (legacy)
For improving the NATTEN patch accuracy. The NATTEN attention replacement is
the **only remaining source of divergence** from the original model. This
branch is for investigating opset 17 LayerNorm, RPB alternatives, or partial
faithful export paths.

---

## What Must NOT Be Mixed

| Cross-contamination | Why |
|---|---|
| Eval scenario changes in `cpp-runtime` | Benchmark integrity — eval must be independent of runtime |
| Patch logic changes in `expanded-eval` | Fidelity claims must not shift under an eval run |
| Model architecture changes in `onnx-deployment` | Would invalidate the existing ONNX/TRT engine cache |
| Deployment claims (latency, speedup) in `faithful-natten` | The patched model ≠ the faithful model |
| C++ code in `expanded-eval` | Runtime work must be independently verifiable |

---

## Suggested Merge Order

```
Phase 1:  expanded-eval   → main  [MERGED March 2026, commit 46792b4]
Phase 2:  cpp-runtime     → main  [MERGED March 2026, commit 93b367b]
Phase 3:  faithful-natten         [legacy — content lives in inference/faithful/;
                                   no formal merge commit; FIDELITY_REPORT.md is the record]

Phase 4 (pending, April 2026 — all branch from inference-extensions):
  4a: feature/nvtx-nsys               → inference-extensions
  4b: feature/reranker-vectorization  → inference-extensions
  4c: feature/int8-ptq                → inference-extensions
  4d: feature/cpp-runtime-optimizations → inference-extensions
  4e: feature/cuda-reranker           → inference-extensions
  then: inference-extensions          → main
```

Merge policy:
```bash
git checkout main
git merge --no-ff expanded-eval   -m "merge: expanded evaluation benchmark v2"
git merge --no-ff cpp-runtime     -m "merge: native C++ ORT + TRT inference"
```

Never fast-forward merge to main — the merge commit is the audit trail.

---

## April 2026 Feature Branches

All five branch from `inference-extensions` (stable base, commit `ce31a6a`).
Each is a single commit with a complete, independently reviewable change.

### `feature/nvtx-nsys` — GPU profiling instrumentation
**Purpose**: Add NVTX range markers so `nsys profile` shows per-stage GPU
timelines. Prerequisite for data-driven optimisation decisions.

Files added/modified:
- `inference/run_pipeline.py` — NVTX ranges around feature build / TRT forward / reranker
- `inference/imitation_planner.py` — NVTX ranges around Python planner stages
- `inference/deploy/benchmark_tensorrt.py` — NVTX ranges in Python TRT benchmark
- `cpp/tensorrt_infer.cpp` — NVTX macros + restructured timed loop
- `cpp/tensorrt_utils.h` — `#define NVTX_*` macros (compile-time enabled via `-DENABLE_NVTX`)
- `script/profile_nsys.sh` — one-shot `nsys profile` launcher
- `.gitignore` — ignore `*.nsys-rep`, `*.sqlite`

### `feature/reranker-vectorization` — Coverage loop vectorisation
**Purpose**: Replace the Python `for k in range(K)` loop in `rerank_modes`
with a single batched `torch.cdist([K*T,2],[R,2])` call. 2.84× CPU speedup,
bit-exact output.

Files modified:
- `src/planners/mode_reranker.py` — vectorised inner loop
- `tests/test_mode_reranker.py` — 3 new `TestVectorizedMatchesLooped` tests
- `benchmarks/bench_reranker.py` — benchmark comparing looped vs vectorised

### `feature/int8-ptq` — TensorRT INT8 post-training quantisation
**Purpose**: Calibrate and benchmark an INT8 TRT engine via entropy
calibration. Adds the full pipeline: calibration data generation, calibrator
class, CLI flags, and validation script updates.

Files added/modified:
- `inference/deploy/dump_calibration_data.py` (new) — generates 512 `.npy`
  calibration batches (23 input tensors each, synthetic constant-fill)
- `inference/deploy/benchmark_tensorrt.py` — `_PlanTFCalibrator`, `--int8`,
  `--calib-dir`, `--calib-cache` CLI args, INT8 benchmark block
- `cpp/validate_outputs.py` — `--int8-engine` arg, INT8 comparison section
- `.gitignore` — `inference/calib_data/`, `*.trt.cache`

### `feature/cpp-runtime-optimizations` — Pinned memory + async DMA + per-stage timers
**Purpose**: Reduce H2D/D2H transfer cost and expose per-stage timing.

Changes:
- `cpp/tensorrt_utils.h` — `EngineBuffer::alloc(bool use_pinned)` uses
  `cudaMallocHost` for page-locked host memory; `free_all()` uses
  `cudaFreeHost`; added `print_latency_stages(h2d, kernel, d2h)` helper
- `cpp/tensorrt_utils.cpp` — `alloc_engine_buffers` gains `bool use_pinned`
  parameter
- `cpp/tensorrt_infer.cpp` — timed loop restructured: 3 × 2 CUDA event
  pairs (H2D / kernel / D2H); `cudaMemcpyAsync` for all transfers;
  single `cudaStreamSynchronize` per iteration; per-stage latency table

### `feature/cuda-reranker` — CUDA kernel for coverage computation
**Purpose**: Replace the Python coverage loop with a fused CUDA kernel,
giving ~3–6× speedup over the CPU PyTorch path.

Files added/modified:
- `src/planners/reranker_cuda_kernel.cu` (new) — kernel layout: Grid(K) ×
  Block(128); each thread handles one waypoint, scans R centres for min
  squared distance; shared-memory tree reduction; BLOCK_T=128 (power-of-two
  required for reduction correctness)
- `src/planners/reranker_cuda.py` (new) — lazy JIT-loader via
  `torch.utils.cpp_extension.load`; returns `None` on failure (no Ninja,
  no GPU) for transparent fallback
- `src/planners/mode_reranker.py` — fast path: `compute_coverage_cuda()` if
  `traj.is_cuda`, else PyTorch loop
- `tests/test_mode_reranker.py` — `TestCudaKernelCoverage` (4 tests, skipped
  when CUDA/Ninja absent); corrected `_cuda_ext_available()` skip guard
- `benchmarks/bench_reranker.py` — 3-way benchmark: CPU loop / GPU loop /
  CUDA kernel (CUDA events, 500 runs)

---

## Guard Rules (enforced by convention, may be scripted)

1. `main` tag must pass `bash script/initial_sanity_check.sh` before merge.
2. `expanded-eval` must include an updated `docs/mini_scenario_inventory.md` before merge.
3. `cpp-runtime` must demonstrate latency parity (within 2×) of Python path before merge.
4. Deployment claims (latency numbers) are only valid against the **patched** ONNX model.
   Claims about the original faithful model require `faithful-natten` work.
