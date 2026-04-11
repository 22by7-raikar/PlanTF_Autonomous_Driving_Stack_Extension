/**
 * reranker_cuda_kernel.cu
 * -----------------------
 * CUDA kernel for planTF mode-reranker coverage computation.
 *
 * What it computes
 * ----------------
 * Given K candidate trajectories (each with T waypoints) and R on-route
 * polygon centres, this kernel computes:
 *
 *   coverage[k] = (number of waypoints in mode k whose minimum Euclidean
 *                  distance to any route centre ≤ threshold) / T
 *
 * That is the same quantity computed by the PyTorch loop in mode_reranker.py;
 * this kernel is just a faster implementation of it.
 *
 * Kernel layout
 * -------------
 *   Grid : (K,)        — one block per trajectory mode
 *   Block: (BLOCK_T,)  — BLOCK_T = 128, which is ≥ T = 80 and a power of two
 *   Shared: BLOCK_T floats — used for the within-block parallel sum reduction
 *
 * Distance arithmetic uses squared distances throughout so that we avoid
 * sqrt on every (t, r) pair.
 *
 * Constraints
 * -----------
 *   T ≤ BLOCK_T   (asserted in the C++ wrapper)
 *   BLOCK_T must be a power of two (required for the reduction loop)
 *
 * Build
 * -----
 * Compiled automatically by torch.utils.cpp_extension.load() from
 * src/planners/reranker_cuda.py on first call.  Cached in
 * ~/.cache/torch_extensions/ (or $TORCH_EXTENSIONS_DIR) thereafter.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block size: must be ≥ T and a power of two for the reduction to work.
// T = 80 for PlanTF → 128 is the smallest valid power-of-two choice.
static constexpr int BLOCK_T = 128;


// ── Kernel ───────────────────────────────────────────────────────────────────
/**
 * coverage_kernel
 *
 * Inputs:
 *   wp        — waypoints [K, T, 2], row-major, float32
 *   rc        — route centres [R, 2], row-major, float32
 *   T, R      — tensor dimensions (K comes from gridDim.x)
 *   thresh_sq — squared distance threshold (ROUTE_DIST_THRESHOLD²)
 *
 * Output:
 *   cov [K]   — pre-zeroed by the caller; each block writes one element
 *
 * Per thread:
 *   Each thread (k, t) loads one waypoint and iterates over all R centres
 *   to find the minimum squared distance.  Distance comparisons are done
 *   in squared space — no sqrt needed.
 *
 * Per block reduction:
 *   After each thread deposits its on_route flag (0 or 1) into shared memory,
 *   a standard tree reduction accumulates the sum.  Thread 0 divides by T
 *   and writes to cov[k].
 */
__global__ void coverage_kernel(
        const float* __restrict__ wp,   // [K, T, 2]
        const float* __restrict__ rc,   // [R, 2]
        float*       __restrict__ cov,  // [K]
        int T, int R, float thresh_sq)
{
    __shared__ float sdata[BLOCK_T];

    const int k = blockIdx.x;   // mode index
    const int t = threadIdx.x;  // waypoint index (may exceed T on pad threads)

    // Threads t < T compute their on_route flag; padded threads contribute 0.
    float on_route = 0.0f;
    if (t < T) {
        const float wx = wp[(k * T + t) * 2 + 0];
        const float wy = wp[(k * T + t) * 2 + 1];

        // Linear scan over all R route centres; find minimum squared distance.
        // R is small (~30–100) so the loop fits well inside L1 cache.
        float min_d2 = 3.402823466e+38f;  // FLT_MAX
        for (int r = 0; r < R; ++r) {
            const float dx = wx - rc[r * 2 + 0];
            const float dy = wy - rc[r * 2 + 1];
            const float d2 = dx * dx + dy * dy;
            if (d2 < min_d2) min_d2 = d2;
        }

        on_route = (min_d2 <= thresh_sq) ? 1.0f : 0.0f;
    }

    // Load into shared memory — padded threads write 0 so they don't bias sum.
    sdata[t] = on_route;
    __syncthreads();

    // Tree reduction: sum all BLOCK_T elements.
    // BLOCK_T must be a power of two for this pattern to be correct.
    for (int s = BLOCK_T >> 1; s > 0; s >>= 1) {
        if (t < s) sdata[t] += sdata[t + s];
        __syncthreads();
    }

    // Thread 0 divides by T (not BLOCK_T — padded threads were 0) and writes.
    if (t == 0) cov[k] = sdata[0] / static_cast<float>(T);
}


// ── C++ wrapper (pybind11 entry point) ───────────────────────────────────────
/**
 * compute_coverage_cuda
 *
 * Called from reranker_cuda.py.  Validates inputs, launches the kernel, and
 * returns coverage as a new [K] float32 tensor on the same CUDA device.
 *
 * Parameters
 * ----------
 * waypoints    : float32 CUDA Tensor, shape [K, T, 2], contiguous
 * route_centers: float32 CUDA Tensor, shape [R, 2],    contiguous
 * threshold    : distance threshold in metres (passed as double, cast to float)
 *
 * Returns
 * -------
 * coverage : float32 CUDA Tensor, shape [K]
 */
torch::Tensor compute_coverage_cuda(
        torch::Tensor waypoints,
        torch::Tensor route_centers,
        double threshold)
{
    TORCH_CHECK(waypoints.is_cuda(),
        "compute_coverage_cuda: waypoints must be a CUDA tensor");
    TORCH_CHECK(route_centers.is_cuda(),
        "compute_coverage_cuda: route_centers must be a CUDA tensor");
    TORCH_CHECK(waypoints.dtype()     == torch::kFloat32,
        "compute_coverage_cuda: waypoints must be float32");
    TORCH_CHECK(route_centers.dtype() == torch::kFloat32,
        "compute_coverage_cuda: route_centers must be float32");
    TORCH_CHECK(waypoints.dim() == 3 && waypoints.size(2) == 2,
        "compute_coverage_cuda: waypoints must have shape [K, T, 2]");
    TORCH_CHECK(route_centers.dim() == 2 && route_centers.size(1) == 2,
        "compute_coverage_cuda: route_centers must have shape [R, 2]");

    const int K = static_cast<int>(waypoints.size(0));
    const int T = static_cast<int>(waypoints.size(1));
    const int R = static_cast<int>(route_centers.size(0));

    TORCH_CHECK(T <= BLOCK_T,
        "compute_coverage_cuda: T=", T,
        " exceeds kernel block size BLOCK_T=", BLOCK_T,
        ". Recompile with a larger BLOCK_T.");

    auto cov = torch::zeros({K}, waypoints.options());
    const float thresh_sq = static_cast<float>(threshold * threshold);

    // Launch K blocks of BLOCK_T threads; shared memory = BLOCK_T floats.
    coverage_kernel<<<K, BLOCK_T, BLOCK_T * sizeof(float)>>>(
        waypoints.data_ptr<float>(),
        route_centers.data_ptr<float>(),
        cov.data_ptr<float>(),
        T, R, thresh_sq);

    // Propagate any kernel launch errors to Python.
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "compute_coverage_cuda: kernel launch failed");

    return cov;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_coverage_cuda", &compute_coverage_cuda,
          "Per-mode on-route coverage kernel for the planTF mode reranker.\n\n"
          "Args:\n"
          "  waypoints     (Tensor[K, T, 2], float32, CUDA)\n"
          "  route_centers (Tensor[R, 2],    float32, CUDA)\n"
          "  threshold     (float) distance threshold in metres\n"
          "Returns:\n"
          "  coverage (Tensor[K], float32, CUDA)");
}
