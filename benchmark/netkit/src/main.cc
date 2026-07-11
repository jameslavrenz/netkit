// netkit MNIST MLP invoke-time benchmark — pairs with benchmark/tflite/mnist_mlp_bench.py.
//
// Uses models/mnist_mlp.nk and the same 10 embedded MNIST test vectors.
// Latency: batch N invokes in one timed window (default 1000) to escape ~1 µs timer noise.
// Accuracy: separate untimed pass over the 10 digit images.
//
// Usage: bench [model.nk] [batch_invokes] [batch_passes]

#include "arena.hpp"
#include "arena_util.hpp"
#include "benchmark_stats.hpp"
#include "mlp.hpp"
#include "mnist_test_images.h"
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

constexpr const char* kDefaultModelPath = "models/mnist_mlp.nk";

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

int ArgMax10(const float* values)
{
    int best = 0;
    float max_val = values[0];
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
            best = i;
        }
    }
    return best;
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

#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity);
#else
    alignas(std::max_align_t) static unsigned char arena_memory[Arena::kDefaultCapacity];
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity, arena_memory);
#endif
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

    if (input_rank != 2 || input_shape[0] != 1 || input_shape[1] != kMnistBenchmarkInputSize)
    {
        std::fprintf(stderr, "unexpected input shape [%u", input_shape[0]);
        for (uint32_t i = 1; i < input_rank; ++i)
            std::fprintf(stderr, ", %u", input_shape[i]);
        std::fprintf(stderr, "]\n");
        return 1;
    }

    const uint32_t output_cols = NkLoader::OutputElements(parsed);
    if (output_cols != 10)
    {
        std::fprintf(stderr, "expected 10 outputs, got %u\n", output_cols);
        return 1;
    }

    Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
    Tensor output = TensorFactory::Create2D(arena, input_shape[0], output_cols);
    if (!input.data || !output.data)
    {
        std::fprintf(stderr, "tensor allocation failed\n");
        return 1;
    }

    std::printf("netkit MNIST MLP benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  model:       %s\n", model_path);
    std::printf("  images:      %d (accuracy only)\n", kMnistBenchmarkImageCount);
    std::printf("  batch:       %d invokes x %d passes (discard pass 0)\n", batch_invokes,
                batch_passes);
    std::printf("  arena bytes: %zu\n", static_cast<size_t>(Arena::kDefaultCapacity));

    float* input_data = static_cast<float*>(input.data);

    int correct = 0;
    for (int i = 0; i < kMnistBenchmarkImageCount; ++i)
    {
        const MnistBenchmarkSample& sample = kMnistBenchmarkImages[i];
        std::memcpy(input_data, sample.pixels, kMnistBenchmarkInputSize * sizeof(float));
        network->forward(input, output, arena);
        const int predicted = ArgMax10(static_cast<const float*>(output.data));
        if (predicted == sample.label)
            ++correct;
        std::printf("  image %d label=%d pred=%d %s\n", i, sample.label, predicted,
                    predicted == sample.label ? "OK" : "MISS");
    }
    std::printf("  accuracy:    %d/%d\n", correct, kMnistBenchmarkImageCount);

    std::memcpy(input_data, kMnistBenchmarkImages[0].pixels,
                kMnistBenchmarkInputSize * sizeof(float));
    network->forward(input, output, arena);

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
        "mlp",
        NETKIT_BENCH_BACKEND,
        BenchmarkStats::Compute(pass_averages_us.data(), batch_passes),
        batch_invokes);

    return correct == kMnistBenchmarkImageCount ? 0 : 1;
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
