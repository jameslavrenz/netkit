// netkit MNIST MLP int8 host benchmark (pairs with benchmark/tflite/mnist_mlp_int8_bench.py).
//
// Loads models/mnist_mlp_int8.nk. Feeds prequantized int8 digits (Python export).
// Latency: batch N invokes in one timed window (default 1000) to escape ~1 µs timer noise.
// Accuracy: separate untimed pass over the 10 digit images.
//
// Usage: bench [model.nk] [batch_invokes] [batch_passes]

#include "arena.hpp"
#include "arena_util.hpp"
#include "benchmark_stats.hpp"
#include "netkit_util.hpp"
#include "mlp.hpp"
#include "mnist_mlp_int8_test_images.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mnist_mlp_int8.nk";
constexpr size_t kArenaCapacity = 4 * 1024 * 1024;

int ParsePositiveInt(const char* s, int fallback)
{
    if (!s || !*s)
        return fallback;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 1 || v > 10000000)
        return fallback;
    return static_cast<int>(v);
}

int RunBenchmark(const char* model_path, int batch_invokes, int batch_passes)
{
    if (batch_passes < 2)
    {
        std::fprintf(stderr, "need at least 2 batch passes to discard cold pass 0\n");
        return 1;
    }
    if (batch_passes > BenchmarkStats::kMaxRuns)
    {
        std::fprintf(stderr, "batch_passes must be <= %d\n", BenchmarkStats::kMaxRuns);
        return 1;
    }

    NkLoader::ParsedModel parsed{};
    const NkLoader::LoadResult parse_result = NkLoader::ParseFile(model_path, parsed);
    if (parse_result.status != NkLoader::LoadStatus::Ok)
    {
        std::fprintf(stderr, "parse failed: %s\n",
                     parse_result.message ? parse_result.message
                                          : NkLoader::StatusMessage(parse_result.status));
        return 1;
    }
    if (parsed.header.network_kind != NkFormat::NetworkKind::Mlp)
    {
        std::fprintf(stderr, "expected MLP model\n");
        return 1;
    }

    alignas(std::max_align_t) static unsigned char arena_memory[kArenaCapacity];
    ArenaUtil::Scoped arena_scope(kArenaCapacity, arena_memory);
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed\n");
        return 1;
    }
    Arena& arena = arena_scope.Get();

    MLPNetwork* network = nullptr;
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    const NkLoader::LoadResult load_result =
        NkLoader::LoadMLP(model_path, arena, network, input_shape, input_rank);
    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        std::fprintf(stderr, "load failed: %s\n",
                     load_result.message ? load_result.message
                                         : NkLoader::StatusMessage(load_result.status));
        return 1;
    }
    if (!network->IsQuantized())
    {
        std::fprintf(stderr, "expected quantized int8 model\n");
        return 1;
    }
    if (input_rank != 2 || input_shape[0] != 1 ||
        input_shape[1] != static_cast<uint32_t>(kMnistMlpInt8BenchmarkInputSize))
    {
        std::fprintf(stderr, "unexpected MLP input shape\n");
        return 1;
    }

    network->SetOmitFinalSoftmax(true);

    alignas(16) int8_t input_buffer[kMnistMlpInt8BenchmarkInputSize] = {};
    alignas(16) int8_t output_buffer[10] = {};
    Tensor input = TensorFactory::View2DInt8(input_buffer, input_shape[0], input_shape[1]);
    Tensor output = TensorFactory::View2DInt8(output_buffer, input_shape[0], 10);

    std::printf("netkit MNIST MLP int8 benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  dtype:       int8\n");
    std::printf("  model:       %s\n", model_path);
    std::printf("  images:      %d (accuracy only)\n", kMnistMlpInt8BenchmarkImageCount);
    std::printf("  batch:       %d invokes x %d passes (discard pass 0)\n", batch_invokes,
                batch_passes);
    std::printf("  arena bytes: %zu\n", kArenaCapacity);

    // Untimed accuracy over the 10 digit images.
    int correct = 0;
    for (int i = 0; i < kMnistMlpInt8BenchmarkImageCount; ++i)
    {
        const MnistMlpInt8BenchmarkSample& sample = kMnistMlpInt8BenchmarkImages[i];
        std::memcpy(input_buffer, sample.pixels,
                    static_cast<size_t>(kMnistMlpInt8BenchmarkInputSize));
        network->forward(input, output, arena);
        if (!output.data || output.type != DataType::Int8)
        {
            std::fprintf(stderr, "forward failed on accuracy image %d (%s)\n", i, sample.name);
            return 1;
        }
        const int predicted = static_cast<int>(
            CmsisQuantUtil::ArgMaxInt8(static_cast<const int8_t*>(output.data), 10u));
        if (predicted == sample.label)
            ++correct;
        std::printf("  image %d label=%d pred=%d %s\n", i, sample.label, predicted,
                    predicted == sample.label ? "OK" : "MISS");
    }
    std::printf("  accuracy:    %d/%d\n", correct, kMnistMlpInt8BenchmarkImageCount);

    // Fixed input for timed batch (image 0) — measures invoke cost, not memcpy.
    std::memcpy(input_buffer, kMnistMlpInt8BenchmarkImages[0].pixels,
                static_cast<size_t>(kMnistMlpInt8BenchmarkInputSize));
    network->forward(input, output, arena);  // bind / warm once before timed passes

    std::array<double, BenchmarkStats::kMaxRuns> pass_averages_us{};
    for (int pass = 0; pass < batch_passes; ++pass)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int n = 0; n < batch_invokes; ++n)
            network->forward(input, output, arena);
        const auto end = std::chrono::steady_clock::now();
        const double window_us =
            std::chrono::duration<double, std::micro>(end - start).count();
        pass_averages_us[static_cast<size_t>(pass)] =
            window_us / static_cast<double>(batch_invokes);
    }

    BenchmarkStats::PrintBatchSummary(
        "netkit",
        "mlp_int8",
        NETKIT_BENCH_BACKEND,
        BenchmarkStats::Compute(pass_averages_us.data(), batch_passes),
        batch_invokes);

    return correct == kMnistMlpInt8BenchmarkImageCount ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    const int batch_invokes =
        ParsePositiveInt(argc > 2 ? argv[2] : nullptr, BenchmarkStats::kDefaultBatchInvokes);
    const int batch_passes =
        ParsePositiveInt(argc > 3 ? argv[3] : nullptr, BenchmarkStats::kDefaultBatchPasses);
    return RunBenchmark(model_path, batch_invokes, batch_passes);
}
