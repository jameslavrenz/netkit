// netkit MNIST MLP invoke-time benchmark — pairs with benchmark/tflm/.
//
// Uses models/mnist_mlp.nk and the same 10 embedded MNIST test vectors as the
// TFLM benchmark. Times only MLPNetwork::forward() with std::chrono.

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
#include <cstring>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mnist_mlp.nk";
constexpr int kRuns = BenchmarkStats::kDefaultRuns;

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

int RunBenchmark(const char* model_path)
{
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
    std::printf("  images:      %d per run\n", kMnistBenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %zu\n", static_cast<size_t>(Arena::kDefaultCapacity));

    std::array<double, BenchmarkStats::kMaxRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistBenchmarkImageCount; ++i)
        {
            const MnistBenchmarkSample& sample = kMnistBenchmarkImages[i];
            float* input_data = static_cast<float*>(input.data);
            std::memcpy(input_data, sample.pixels, kMnistBenchmarkInputSize * sizeof(float));

            const auto start = std::chrono::steady_clock::now();
            network->forward(input, output, arena);
            const auto end = std::chrono::steady_clock::now();

            const double elapsed_us =
                std::chrono::duration<double, std::micro>(end - start).count();

            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const float* output_data = static_cast<const float*>(output.data);
                const int predicted = ArgMax10(output_data);
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    std::printf("  accuracy:    %d/%d on final run\n", correct, kMnistBenchmarkImageCount);
    BenchmarkStats::PrintSummary("netkit", "mlp", NETKIT_BENCH_BACKEND,
                                 BenchmarkStats::Compute(run_averages_us.data(), kRuns));

    return correct == kMnistBenchmarkImageCount ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
