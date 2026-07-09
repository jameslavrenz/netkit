// netkit MobileNetV4-small float32 invoke-time benchmark (host/CPU).
//
// Loads models/mobilenetv4_small.nk (CNN, input [56,56,3], 10 classes) — a
// depthwise-heavy model that exercises ConvDepthwiseForward. Feeds 10 MNIST
// images (one per class) upsampled 28x28x1 -> 56x56x3, loops over them kLoops
// times, and times every CNNNetwork::forward() invoke with std::chrono.
//
// Reports the cold first invoke separately plus the warm mean/min/max/stddev.

#include "arena.hpp"
#include "arena_util.hpp"
#include "cnn.hpp"
#include "cmsis_dsp_util.hpp"
#include "mnist_cnn_test_images.h"
#include "mobilenetv4_netkit_int8_test_images.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

#ifndef NETKIT_DW_ROW_ACCUM
#define NETKIT_DW_ROW_ACCUM 1
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mobilenetv4_small.nk";
constexpr int kLoops = 30;   // passes over the 10 images -> 300 invokes total
constexpr uint32_t kInH = 56;
constexpr uint32_t kInW = 56;
constexpr uint32_t kInC = 3;
constexpr size_t kArenaCapacity = 64 * 1024 * 1024;

std::size_t ArenaBytesForModel(const NkLoader::ParsedModel& parsed)
{
    return ArenaUtil::CapacityForModel(NkLoader::InputElements(parsed),
                                      parsed.header.network_kind == NkFormat::NetworkKind::Cnn,
                                      parsed.header.weights_bytes,
                                      parsed.header.biases_bytes);
}

int ArgMax(const float* values, int count)
{
    int best = 0;
    float max_val = values[0];
    for (int i = 1; i < count; ++i)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
            best = i;
        }
    }
    return best;
}

// Upsample a 28x28x1 MNIST image (row-major) into a 56x56x3 NHWC buffer by
// 2x nearest-neighbour and channel replication. Content is irrelevant to timing;
// this just gives MobileNetV4 a valid, MNIST-derived input of the right shape.
void UpsampleMnistTo56x56x3(const float* src28, float* dst)
{
    for (uint32_t y = 0; y < kInH; ++y)
    {
        for (uint32_t x = 0; x < kInW; ++x)
        {
            const float v = src28[(y / 2) * 28 + (x / 2)];
            float* px = dst + (y * kInW + x) * kInC;
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
    }
}

Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
{
    const std::array<uint32_t, 3> shape{h, w, c};
    return TensorFactory::ViewND(data, 3, std::span<const uint32_t>(shape));
}

Tensor MakeNhwcViewInt8(int8_t* data, uint32_t h, uint32_t w, uint32_t c)
{
    Tensor input{};
    input.data = data;
    input.type = DataType::Int8;
    input.rank = 3;
    input.shape[0] = h;
    input.shape[1] = w;
    input.shape[2] = c;
    input.stride[0] = w * c;
    input.stride[1] = c;
    input.stride[2] = 1;
    input.num_elements = h * w * c;
    input.bytes = input.num_elements * sizeof(int8_t);
    return input;
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
    if (parsed.header.network_kind != NkFormat::NetworkKind::Cnn)
    {
        std::fprintf(stderr, "expected CNN model\n");
        return 1;
    }

    // Int8 plan activations need a large host arena; floor at kArenaCapacity.
    const std::size_t arena_capacity =
        std::max(ArenaBytesForModel(parsed), kArenaCapacity);
#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(arena_capacity, nullptr);
#else
    static unsigned char* arena_memory = new unsigned char[arena_capacity];
    ArenaUtil::Scoped arena_scope(arena_capacity, arena_memory);
#endif
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed\n");
        return 1;
    }
    Arena& arena = arena_scope.Get();

    CNNNetwork* network = nullptr;
    std::array<uint32_t, kMaxTensorRank> input_shape{};
    uint32_t input_rank = 0;
    const NkLoader::LoadResult load_result =
        NkLoader::LoadCNN(model_path, arena, network, input_shape, input_rank);
    if (load_result.status != NkLoader::LoadStatus::Ok || !network || !network->IsValid())
    {
        std::fprintf(stderr, "load failed: %s\n",
                     load_result.message ? load_result.message
                                         : NkLoader::StatusMessage(load_result.status));
        return 1;
    }

    if (input_rank != 3 || input_shape[0] != kInH || input_shape[1] != kInW ||
        input_shape[2] != kInC)
    {
        std::fprintf(stderr, "unexpected input shape [%u,%u,%u], expected [%u,%u,%u]\n",
                     input_shape[0], input_shape[1], input_shape[2], kInH, kInW, kInC);
        return 1;
    }

    const uint32_t output_cols = NkLoader::OutputElements(parsed);
    const bool quantized = network->IsQuantized();
    const char* dtype = quantized ? "int8" : "float32";

    // Float: upsample MNIST float fixtures. Int8: prequantized Python export (no C++ quant).
    const int num_images =
        quantized ? kMobilenetV4Int8BenchmarkImageCount : kMnistCnnBenchmarkImageCount;
    std::vector<std::vector<float>> float_inputs;
    std::vector<std::vector<int8_t>> int8_inputs;
    if (quantized)
    {
        if (kMobilenetV4Int8BenchmarkInputSize !=
            static_cast<int>(kInH * kInW * kInC))
        {
            std::fprintf(stderr, "int8 fixture size mismatch\n");
            return 1;
        }
        int8_inputs.resize(static_cast<size_t>(num_images));
        for (int i = 0; i < num_images; ++i)
        {
            int8_inputs[static_cast<size_t>(i)].assign(
                kMobilenetV4Int8BenchmarkImages[i].pixels,
                kMobilenetV4Int8BenchmarkImages[i].pixels + kMobilenetV4Int8BenchmarkInputSize);
        }
    }
    else
    {
        float_inputs.resize(static_cast<size_t>(num_images));
        for (int i = 0; i < num_images; ++i)
        {
            float_inputs[static_cast<size_t>(i)].resize(kInH * kInW * kInC);
            UpsampleMnistTo56x56x3(kMnistCnnBenchmarkImages[i].pixels,
                                   float_inputs[static_cast<size_t>(i)].data());
        }
    }

    const char* dw_mode = (NETKIT_DW_ROW_ACCUM ? "row-accum(4)" : "serial");
    std::printf("netkit MobileNetV4-small benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  depthwise:   %s\n", dw_mode);
    std::printf("  dtype:       %s\n", dtype);
    std::printf("  model:       %s\n", model_path);
    std::printf("  input:       %ux%ux%u  outputs: %u\n", kInH, kInW, kInC, output_cols);
    std::printf("  method:      %d images x %d loops = %d invokes (all timed)\n", num_images,
                kLoops, num_images * kLoops);

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(num_images * kLoops));

    int last_pred = -1;
    for (int loop = 0; loop < kLoops; ++loop)
    {
        for (int i = 0; i < num_images; ++i)
        {
            Tensor input =
                quantized
                    ? MakeNhwcViewInt8(int8_inputs[static_cast<size_t>(i)].data(), kInH, kInW, kInC)
                    : MakeNhwcView(float_inputs[static_cast<size_t>(i)].data(), kInH, kInW, kInC);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data)
            {
                std::fprintf(stderr, "forward failed on loop %d image %d\n", loop, i);
                return 1;
            }
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            if (quantized)
            {
                if (output.type != DataType::Int8)
                {
                    std::fprintf(stderr, "expected int8 output\n");
                    return 1;
                }
                last_pred = static_cast<int>(CmsisQuantUtil::ArgMaxInt8(
                    static_cast<const int8_t*>(output.data), output_cols));
            }
            else
            {
                last_pred = ArgMax(static_cast<const float*>(output.data),
                                   static_cast<int>(output_cols));
            }
        }
    }

    // Stats: cold first invoke vs warm (invokes 2..N).
    const double cold_us = samples.front();
    double warm_sum = 0.0;
    double warm_min = samples[1];
    double warm_max = samples[1];
    const size_t warm_n = samples.size() - 1;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        warm_sum += samples[k];
        if (samples[k] < warm_min) warm_min = samples[k];
        if (samples[k] > warm_max) warm_max = samples[k];
    }
    const double warm_mean = warm_sum / static_cast<double>(warm_n);
    double var = 0.0;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        const double d = samples[k] - warm_mean;
        var += d * d;
    }
    const double warm_std = std::sqrt(var / static_cast<double>(warm_n));

    // Median of warm samples (noise-robust; host has background contention).
    std::vector<double> warm_sorted(samples.begin() + 1, samples.end());
    std::sort(warm_sorted.begin(), warm_sorted.end());
    const double warm_median = warm_sorted[warm_sorted.size() / 2];

    std::printf("  last pred:   class %d (fixture weights; accuracy not meaningful)\n", last_pred);
    std::printf("\n");
    std::printf("netkit MobileNetV4-small benchmark summary (%s, depthwise=%s)\n",
                NETKIT_BENCH_BACKEND, dw_mode);
    std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
    std::printf("  warm median:      %9.3f us (%7.3f ms)  <- primary metric\n", warm_median,
                warm_median / 1000.0);
    std::printf("  warm min:         %9.3f us (%7.3f ms)\n", warm_min, warm_min / 1000.0);
    std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n", warm_mean,
                warm_mean / 1000.0, warm_n);
    std::printf("  warm max:         %9.3f us\n", warm_max);
    std::printf("  warm stddev:      %9.3f us\n", warm_std);
    std::printf("BENCHMARK_SUMMARY runtime=netkit model=mobilenetv4_small dtype=%s backend=%s depthwise=%s "
                "warm_median_us=%.3f warm_min_us=%.3f warm_mean_us=%.3f cold_us=%.3f invokes=%zu\n",
                dtype, NETKIT_BENCH_BACKEND, dw_mode, warm_median, warm_min, warm_mean, cold_us,
                samples.size());

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
