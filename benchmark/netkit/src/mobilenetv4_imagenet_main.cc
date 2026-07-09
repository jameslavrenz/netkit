// netkit MobileNetV4-Conv-Small float32 ImageNet host benchmark.
//
// Loads models/mobilenetv4_imagenet_f32.nk (224x224x3, 1000 classes, pretrained
// timm weights). Runs 10 ImageNet-preprocessed images (10 distinct classes),
// reports top-1 accuracy and mean inference time over those 10 invokes.

#include "arena.hpp"
#include "arena_util.hpp"
#include "cnn.hpp"
#include "imagenet_mnv4_test_images.h"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#ifndef NETKIT_BENCH_BACKEND
#define NETKIT_BENCH_BACKEND "reference"
#endif

#ifndef NETKIT_DW_ROW_ACCUM
#define NETKIT_DW_ROW_ACCUM 1
#endif

namespace {

constexpr const char* kDefaultModelPath = "models/mobilenetv4_imagenet_f32.nk";
// One pass over the 10 images is the primary accuracy/latency sample; extra loops
// improve warm timing stability without changing the accuracy report.
constexpr int kLoops = 5;
constexpr uint32_t kInH = kImagenetMnv4BenchmarkHeight;
constexpr uint32_t kInW = kImagenetMnv4BenchmarkWidth;
constexpr uint32_t kInC = kImagenetMnv4BenchmarkChannels;
// Generous host arena: ImageNet 224 activations + scratch need far more than the
// default 64 KiB / even the 64 MiB 56x56 tier. 256 MiB is ample headroom.
constexpr size_t kArenaCapacity = 256 * 1024 * 1024;

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

Tensor MakeNhwcView(float* data, uint32_t h, uint32_t w, uint32_t c)
{
    const std::array<uint32_t, 3> shape{h, w, c};
    return TensorFactory::ViewND(data, 3, std::span<const uint32_t>(shape));
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

#if defined(NETKIT_ARENA_HEAP)
    ArenaUtil::Scoped arena_scope(kArenaCapacity, nullptr);
#else
    static unsigned char* arena_memory = new unsigned char[kArenaCapacity];
    ArenaUtil::Scoped arena_scope(kArenaCapacity, arena_memory);
#endif
    if (!arena_scope)
    {
        std::fprintf(stderr, "arena init failed (requested %zu bytes)\n", kArenaCapacity);
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

    if (network->IsQuantized())
    {
        std::fprintf(stderr,
                     "quantized model requires mobilenetv4_imagenet_int8_main "
                     "(prequantized int8 I/O)\n");
        return 1;
    }

    const uint32_t output_cols = NkLoader::OutputElements(parsed);
    if (output_cols != static_cast<uint32_t>(kImagenetMnv4BenchmarkNumClasses))
    {
        std::fprintf(stderr, "unexpected output count %u (want %d)\n", output_cols,
                     kImagenetMnv4BenchmarkNumClasses);
        return 1;
    }

    const char* dtype = "float32";
    const int num_images = kImagenetMnv4BenchmarkImageCount;
    const char* dw_mode = (NETKIT_DW_ROW_ACCUM ? "row-accum(4)" : "serial");
    std::printf("netkit MobileNetV4 ImageNet benchmark\n");
    std::printf("  backend:     %s\n", NETKIT_BENCH_BACKEND);
    std::printf("  depthwise:   %s\n", dw_mode);
    std::printf("  dtype:       %s\n", dtype);
    std::printf("  model:       %s\n", model_path);
    std::printf("  input:       %ux%ux%u  outputs: %u\n", kInH, kInW, kInC, output_cols);
    std::printf("  arena:       %zu bytes\n", kArenaCapacity);
    std::printf("  method:      %d images x %d loops = %d invokes (all timed)\n", num_images,
                kLoops, num_images * kLoops);
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK
    std::printf("  note:        XNNPACK LayerFast + CMSIS-DSP VectorFast; flags=tflite (-O3)\n");
#else
    std::printf("  note:        flags=tflite (-O3, SIMD on) to match TF Lite / LiteRT MPU\n");
#endif

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(num_images * kLoops));
    std::vector<int> first_pass_preds(static_cast<size_t>(num_images), -1);
    int correct = 0;

    for (int loop = 0; loop < kLoops; ++loop)
    {
        for (int i = 0; i < num_images; ++i)
        {
            // Copy into a mutable buffer (fixture pixels are const).
            std::vector<float> input_buf(
                kImagenetMnv4BenchmarkImages[i].pixels,
                kImagenetMnv4BenchmarkImages[i].pixels + kImagenetMnv4BenchmarkInputSize);
            Tensor input = MakeNhwcView(input_buf.data(), kInH, kInW, kInC);

            const auto start = std::chrono::steady_clock::now();
            Tensor& output = network->forward(input, arena);
            const auto end = std::chrono::steady_clock::now();

            if (!output.data)
            {
                std::fprintf(stderr, "forward failed on loop %d image %d\n", loop, i);
                return 1;
            }
            samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
            const int pred = ArgMax(static_cast<const float*>(output.data),
                                    static_cast<int>(output_cols));
            if (loop == 0)
            {
                first_pass_preds[static_cast<size_t>(i)] = pred;
                if (pred == kImagenetMnv4BenchmarkImages[i].label)
                    ++correct;
                std::printf("  image %d %-28s label=%4d pred=%4d %s\n", i,
                            kImagenetMnv4BenchmarkImages[i].name,
                            kImagenetMnv4BenchmarkImages[i].label, pred,
                            pred == kImagenetMnv4BenchmarkImages[i].label ? "OK" : "MISS");
            }
        }
    }

    const double cold_us = samples.front();
    double all_sum = 0.0;
    for (double s : samples)
        all_sum += s;
    const double all_mean = all_sum / static_cast<double>(samples.size());

    // Mean over the first pass only (10 images) — primary latency for the set.
    double first_pass_sum = 0.0;
    for (int i = 0; i < num_images; ++i)
        first_pass_sum += samples[static_cast<size_t>(i)];
    const double first_pass_mean = first_pass_sum / static_cast<double>(num_images);

    double warm_sum = 0.0;
    double warm_min = samples[1];
    double warm_max = samples[1];
    const size_t warm_n = samples.size() - 1;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        warm_sum += samples[k];
        if (samples[k] < warm_min)
            warm_min = samples[k];
        if (samples[k] > warm_max)
            warm_max = samples[k];
    }
    const double warm_mean = warm_sum / static_cast<double>(warm_n);
    double var = 0.0;
    for (size_t k = 1; k < samples.size(); ++k)
    {
        const double d = samples[k] - warm_mean;
        var += d * d;
    }
    const double warm_std = std::sqrt(var / static_cast<double>(warm_n));
    std::vector<double> warm_sorted(samples.begin() + 1, samples.end());
    std::sort(warm_sorted.begin(), warm_sorted.end());
    const double warm_median = warm_sorted[warm_sorted.size() / 2];

    const double top1 = 100.0 * static_cast<double>(correct) / static_cast<double>(num_images);
    std::printf("\n");
    std::printf("netkit MobileNetV4 ImageNet summary (%s)\n", NETKIT_BENCH_BACKEND);
    std::printf("  top-1 accuracy:   %d / %d  (%.1f%%)\n", correct, num_images, top1);
    std::printf("  10-image mean:    %9.3f us (%7.3f ms)  <- primary latency\n", first_pass_mean,
                first_pass_mean / 1000.0);
    std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
    std::printf("  warm median:      %9.3f us (%7.3f ms)\n", warm_median, warm_median / 1000.0);
    std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n", warm_mean,
                warm_mean / 1000.0, warm_n);
    std::printf("  warm min/max:     %9.3f / %.3f us\n", warm_min, warm_max);
    std::printf("  warm stddev:      %9.3f us\n", warm_std);
    std::printf("  all-invoke mean:  %9.3f us\n", all_mean);
    std::printf(
        "BENCHMARK_SUMMARY runtime=netkit model=mobilenetv4_imagenet dtype=%s backend=%s "
        "top1_correct=%d top1_total=%d top1_pct=%.1f ten_image_mean_us=%.3f warm_median_us=%.3f "
        "warm_mean_us=%.3f cold_us=%.3f invokes=%zu\n",
        dtype, NETKIT_BENCH_BACKEND, correct, num_images, top1, first_pass_mean, warm_median,
        warm_mean, cold_us, samples.size());

    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* model_path = (argc > 1) ? argv[1] : kDefaultModelPath;
    return RunBenchmark(model_path);
}
