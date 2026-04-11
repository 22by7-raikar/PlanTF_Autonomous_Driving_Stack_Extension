/**
 * tensorrt_infer.cpp
 * ------------------
 * planTF TensorRT C++ inference benchmark.
 *
 * Execution pipeline:
 *   1. Load engine cache (inference/planTF.trt or planTF_fp32.trt) if present.
 *   2. If cache absent or --rebuild: parse inference/planTF.onnx and build
 *      a new engine (slow, ~1–5 min on first build).
 *   3. Allocate I/O buffers via alloc_engine_buffers(use_pinned=true):
 *      page-locked host memory (cudaMallocHost) is required for async DMA.
 *      Inputs are constant-filled (float=0.1, int=0, bool=false).
 *   4. Run warmup iterations (kernel only), then timed iterations.
 *      Each timed iteration records 3 CUDA event pairs around:
 *        Stage 1: H2D async transfer of all input tensors
 *        Stage 2: enqueueV3 (TRT kernel)
 *        Stage 3: D2H async transfer of all output tensors
 *      cudaStreamSynchronize is called once per iteration.
 *   5. Print kernel latency table (mean/p50/p95/p99/max/QPS) and
 *      per-stage breakdown (H2D / Kernel / D2H / Total).
 *
 * Expected output (RTX 3060, batch=1):
 *   FP16 engine: mean ~1.1 ms, QPS ~900
 *   FP32 engine: mean ~1.6 ms, QPS ~625
 *
 * Comparison:
 *   FP32 engine ≈ ORT output within ~1.6 mm trajectory deviation.
 *   FP16 engine may diverge ~59 mm (LayerNorm overflow; see CPP_RUNTIME_REPORT.md).
 *   Python TRT ref: conda run -n plantf python inference/deploy/benchmark_tensorrt.py
 *   C++ ORT ref:    ./ort_infer --model ../../inference/planTF.onnx
 *
 * Build:
 *   cmake .. -DENABLE_CUDA=ON -DENABLE_TRT=ON
 *   make -j$(nproc)
 *
 * Run:
 *   ./trt_infer                                           # FP16 cached engine
 *   ./trt_infer --engine ../../inference/planTF_fp32.trt # FP32 cached engine
 *   ./trt_infer --rebuild                                 # rebuild from ONNX
 *
 * Requires: TensorRT 10.x, CUDA 11.8+, tensorrt_utils.h/.cpp
 */

#ifdef ENABLE_TRT

#include "tensorrt_utils.h"

#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ── Args ────────────────────────────────────────────────────────────────────
struct Args {
    std::string engine_path = "../../inference/planTF.trt";
    std::string onnx_path   = "../../inference/planTF.onnx";
    int warmup = 10, runs = 50;
    bool fp32 = false;
    bool rebuild = false;
    bool verbose = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--engine"  && i+1 < argc) { a.engine_path = argv[++i]; }
        else if (arg == "--onnx"    && i+1 < argc) { a.onnx_path   = argv[++i]; }
        else if (arg == "--warmup"  && i+1 < argc) { a.warmup = std::stoi(argv[++i]); }
        else if (arg == "--runs"    && i+1 < argc) { a.runs = std::stoi(argv[++i]); }
        else if (arg == "--fp32")    { a.fp32 = true; }
        else if (arg == "--rebuild") { a.rebuild = true; }
        else if (arg == "--verbose") { a.verbose = true; }
        else if (arg == "--help") {
            std::cout <<
                "Usage: trt_infer [options]\n"
                "  --engine PATH   Engine cache path  (default: ../../inference/planTF.trt)\n"
                "  --onnx   PATH   ONNX model path    (default: ../../inference/planTF.onnx)\n"
                "  --warmup N      Warmup runs        (default: 10)\n"
                "  --runs   N      Timed runs         (default: 50)\n"
                "  --fp32          Force FP32 engine  (default: FP16 if supported)\n"
                "  --rebuild       Rebuild engine even if cache exists\n"
                "  --verbose       Print tensor shapes\n";
            std::exit(0);
        }
    }
    return a;
}

// ── Build or load engine ────────────────────────────────────────────────────
static std::vector<char> load_engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    auto size = f.tellg(); f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

static std::vector<char> build_engine(
        const std::string& onnx_path,
        bool fp32,
        TRTLogger& logger) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("createInferBuilder failed");

    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger));

    std::cout << "  Parsing ONNX: " << onnx_path << " ..." << std::flush;
    if (!parser->parseFromFile(onnx_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("ONNX parse failed");
    }
    std::cout << " OK\n";

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    if (!fp32 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "  Precision: FP16\n";
    } else {
        std::cout << "  Precision: FP32\n";
    }

    std::cout << "  Building engine (may take 1-5 min) ..." << std::flush;
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("Engine build failed");
    std::cout << " OK\n";

    return std::vector<char>(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());
}



// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    std::cout << "\n============================================================\n";
    std::cout << "  planTF TensorRT C++ Inference\n";
    std::cout << "============================================================\n";
    std::cout << "  Engine : " << args.engine_path << "\n";
    std::cout << "  ONNX   : " << args.onnx_path   << "\n";
    std::cout << "  Warmup : " << args.warmup << "   Runs: " << args.runs << "\n\n";

    TRTLogger logger(nvinfer1::ILogger::Severity::kWARNING);

    // ── Load or build engine ─────────────────────────────────────────────────
    std::vector<char> engine_data;
    bool cache_exists = static_cast<bool>(std::ifstream(args.engine_path));

    if (!args.rebuild && cache_exists) {
        std::cout << "  Loading cached engine: " << args.engine_path << "\n";
        engine_data = load_engine(args.engine_path);
    } else {
        std::cout << "  Building engine from ONNX...\n";
        engine_data = build_engine(args.onnx_path, args.fp32, logger);
        // Save cache
        std::ofstream f(args.engine_path, std::ios::binary);
        f.write(engine_data.data(), engine_data.size());
        std::cout << "  Saved engine: " << args.engine_path << "\n";
    }

    // ── Deserialize ─────────────────────────────────────────────────────────
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger));
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) throw std::runtime_error("Deserialize engine failed");

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext());
    if (!context) throw std::runtime_error("createExecutionContext failed");

    // ── Allocate I/O buffers (pinned host memory for async H2D/D2H) ────────────
    // use_pinned=true: cudaMallocHost gives page-locked host memory.
    // This is required for cudaMemcpyAsync to actually be asynchronous;
    // pageable memory forces CUDA to stage through a locked bounce buffer
    // anyway (synchronously), so pinned memory also improves raw bandwidth.
    std::cout << "  IO tensors: " << engine->getNbIOTensors() << "\n";
    auto buffers = alloc_engine_buffers(engine.get(), args.verbose, /*use_pinned=*/true);
    for (auto& buf : buffers)
        context->setTensorAddress(buf.name.c_str(), buf.device_ptr);

    // ── CUDA stream + per-stage events ──────────────────────────────────────
    // We create 3 event pairs around each pipeline stage so we can break
    // down total latency into H2D transfer / TRT kernel / D2H transfer.
    // cudaEventRecord enqueues the timestamp into the stream — events are
    // measured entirely on the GPU side with ~2 µs resolution.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t ev_h2d_s, ev_h2d_e;  // host-to-device transfer
    cudaEvent_t ev_ker_s,  ev_ker_e;  // TRT enqueueV3 kernel
    cudaEvent_t ev_d2h_s,  ev_d2h_e;  // device-to-host transfer
    CUDA_CHECK(cudaEventCreate(&ev_h2d_s));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_e));
    CUDA_CHECK(cudaEventCreate(&ev_ker_s));
    CUDA_CHECK(cudaEventCreate(&ev_ker_e));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_s));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_e));

    // ── Warmup ───────────────────────────────────────────────────────────────
    std::cout << "\nWarmup (" << args.warmup << " runs)..." << std::flush;
    for (int i = 0; i < args.warmup; ++i) {
        context->enqueueV3(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::cout << " done\n";

    // ── Timed runs ───────────────────────────────────────────────────────────
    // Each iteration records 3 event pairs around:
    //   Stage 1: H2D — async copy of all input tensors into the stream
    //   Stage 2: Kernel — enqueueV3 submits all TRT layers into the stream
    //   Stage 3: D2H — async copy of all output tensors back to host
    // After all three submits we call cudaStreamSynchronize once so the
    // GPU only stalls the CPU at the very end of the pipeline, not between
    // stages.  cudaEventElapsedTime then reads the GPU timestamps.
    std::cout << "Timing (" << args.runs << " runs, CUDA events, per-stage)..." << std::flush;
    std::vector<double> times_h2d, times_ker, times_d2h;
    times_h2d.reserve(args.runs);
    times_ker.reserve(args.runs);
    times_d2h.reserve(args.runs);

    for (int i = 0; i < args.runs; ++i) {
        // Stage 1: H2D (async, inputs only)
        // cudaMemcpyAsync returns immediately; the DMA engine handles the
        // transfer in the background while the CPU records the next event.
        CUDA_CHECK(cudaEventRecord(ev_h2d_s, stream));
        for (auto& buf : buffers) {
            if (!buf.is_input()) continue;
            CUDA_CHECK(cudaMemcpyAsync(
                buf.device_ptr, buf.host_ptr,
                static_cast<size_t>(buf.nbytes),
                cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaEventRecord(ev_h2d_e, stream));

        // Stage 2: TRT kernel
        // enqueueV3 appends all network layers to the stream — they will
        // execute after all H2D transfers in this stream complete.
        CUDA_CHECK(cudaEventRecord(ev_ker_s, stream));
        context->enqueueV3(stream);
        CUDA_CHECK(cudaEventRecord(ev_ker_e, stream));

        // Stage 3: D2H (async, outputs only)
        // Similarly, these launches are serialised in-stream so they wait
        // for the kernel to finish before starting transfer.
        CUDA_CHECK(cudaEventRecord(ev_d2h_s, stream));
        for (auto& buf : buffers) {
            if (!buf.is_output()) continue;
            CUDA_CHECK(cudaMemcpyAsync(
                buf.host_ptr, buf.device_ptr,
                static_cast<size_t>(buf.nbytes),
                cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaEventRecord(ev_d2h_e, stream));

        // Single sync point — CPU waits only here, not between stages.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        times_h2d.push_back(static_cast<double>(cuda_event_ms(ev_h2d_s, ev_h2d_e)));
        times_ker.push_back(static_cast<double>(cuda_event_ms(ev_ker_s,  ev_ker_e)));
        times_d2h.push_back(static_cast<double>(cuda_event_ms(ev_d2h_s,  ev_d2h_e)));
    }
    std::cout << " done\n";

    // ── Results ──────────────────────────────────────────────────────────────
    std::string label = args.fp32 ? "TRT FP32" : "TRT FP16";
    print_latency(times_ker, label + " kernel");
    print_latency_stages(times_h2d, times_ker, times_d2h);

    // ── Print output tensor values ────────────────────────────────────────────
    // D2H was performed inside the timed loop, so host_ptr already holds
    // the final output — no extra memcpy needed here.
    std::cout << "\nOutput tensors:\n";
    for (auto& buf : buffers) {
        if (!buf.is_output()) continue;
        std::cout << "  " << buf.name << "  shape=[";
        for (size_t d = 0; d < buf.shape.size(); ++d)
            std::cout << (d?",":"") << buf.shape[d];
        std::cout << "]\n  first 6 values: ";
        const float* p = static_cast<const float*>(buf.host_ptr);
        for (int j = 0; j < 6; ++j)
            std::cout << (j?", ":"") << p[j];
        std::cout << "\n";
    }

    // ── Comparison hint ────────────────────────────────────────────────────
    std::cout << "\nComparison:\n";
    std::cout << "  Python TRT: conda run -n plantf python"
                 " inference/deploy/benchmark_tensorrt.py\n";
    std::cout << "  Python ORT: conda run -n plantf python"
                 " inference/deploy/benchmark_latency.py\n";
    std::cout << "  C++ ORT:    ./ort_infer --model ../../inference/planTF.onnx\n\n";
    std::cout << "  Expected: FP32 engine ≈ ORT (max diff < 2mm)\n";
    std::cout << "            FP16 may diverge ~59mm (see CPP_RUNTIME_REPORT.md)\n";

    // ── Cleanup ────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(ev_h2d_s));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_e));
    CUDA_CHECK(cudaEventDestroy(ev_ker_s));
    CUDA_CHECK(cudaEventDestroy(ev_ker_e));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_s));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_e));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (auto& buf : buffers) buf.free_all();

    return 0;
}

#else  // ENABLE_TRT not set

int main() {
    std::cerr << "ERROR: This binary requires TensorRT.\n"
              << "  Rebuild: cmake .. -DENABLE_CUDA=ON -DENABLE_TRT=ON\n";
    return 1;
}

#endif  // ENABLE_TRT
