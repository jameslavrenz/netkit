// Seeed XIAO ESP32S3 — MNIST CNN float32 (netkit reference, lowered AOT).
// Methodology: 10 runs × 10 images, discard image 0 each run. ESP-NN has no float API.
// Embed (--no-lower) mispredicts on S3 (KI-001); keep lowered for published peers.

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "mnist_cnn_aot.hpp"
#include "mnist_cnn_test_images.h"
#include "netkit_config.h"
#include "netkit_util.hpp"

#include <array>
#include <cstdio>
#include <cstring>

namespace aot = netkit::aot::mnist_cnn;

namespace {

constexpr int kRuns = 10;
constexpr int kImageCount = kMnistCnnBenchmarkImageCount;
constexpr int kInputSize = kMnistCnnBenchmarkInputSize;
constexpr int kOutputClasses = 10;
constexpr std::size_t kArenaCapacity = 256u * 1024u;

static unsigned char* g_arena_memory = nullptr;
alignas(std::max_align_t) static float g_output[aot::kOutputElements];
alignas(std::max_align_t) static float g_input_staging[kInputSize];

int ArgMax10(const float* values)
{
    return static_cast<int>(NetkitUtil::ArgMaxF32(values, kOutputClasses));
}

void CopyInputF32(float* dst, const float* src)
{
    NetkitUtil::CopyF32(src, dst, static_cast<uint32_t>(kInputSize));
}

void PrintOutF32(const float* values)
{
    for (int i = 0; i < kOutputClasses; ++i)
    {
        std::printf("%s%.4f", i ? "," : "", static_cast<double>(values[i]));
    }
}

void PrintDigitSummary(int image, int label, int predicted, float pred_f, const float* out, int ok)
{
    std::printf(
        "DIGIT_SUMMARY runtime=netkit model=cnn_f32 image=%d label=%d pred=%d pred_f=%.4f ok=%d out_f=",
        image, label, predicted, static_cast<double>(pred_f), ok);
    PrintOutF32(out);
    std::printf("\n");
}

unsigned char* AllocArena(std::size_t bytes)
{
    unsigned char* p = static_cast<unsigned char*>(
        heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (p == nullptr)
    {
        p = static_cast<unsigned char*>(
            heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    }
    return p;
}

}  // namespace

extern "C" void app_main(void)
{
    std::printf("\nnetkit XIAO ESP32S3 MNIST CNN float32 benchmark\n");
    std::printf("  backend:     netkit reference (MCU ESP32S3 FPU, lowered AOT)\n");
    std::printf("  weights:     flash (lowered static arrays)\n");
    std::printf("  dtype:       float32 end-to-end\n");
    std::printf("  classify:    argmax(logits) — final Softmax omitted\n");
    std::printf("  images:      %d per run\n", kImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %u\n", static_cast<unsigned>(kArenaCapacity));
    std::printf("  sysclk:      240000000 Hz\n");

    g_arena_memory = AllocArena(kArenaCapacity);
    if (g_arena_memory == nullptr)
    {
        std::printf("ERR arena alloc\n");
        return;
    }

    Arena arena;
    if (!aot::InitArena(arena, g_arena_memory, kArenaCapacity))
    {
        std::printf("ERR arena init (need %u)\n",
                    static_cast<unsigned>(aot::kArenaBytesRecommended));
        return;
    }

    aot::Model model;
    if (!model.load(arena))
    {
        std::printf("ERR model load\n");
        return;
    }
    std::printf("  model:       loaded\n");

    {
        const MnistCnnBenchmarkSample& probe = kMnistCnnBenchmarkImages[0];
        CopyInputF32(g_input_staging, probe.pixels);
        if (!model.forward(arena, g_input_staging, g_output))
        {
            std::printf("ERR probe forward\n");
            return;
        }
        const int predicted = ArgMax10(g_output);
        std::printf("  probe:       label=%d pred=%d pred_f=%.4f out_f=",
                    probe.label, predicted, static_cast<double>(g_output[predicted]));
        PrintOutF32(g_output);
        std::printf("\n");
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kImageCount> final_predictions{};
    std::array<float, kImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;
        for (int i = 0; i < kImageCount; ++i)
        {
            const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];
            CopyInputF32(g_input_staging, sample.pixels);
            const int64_t start_us = esp_timer_get_time();
            if (!model.forward(arena, g_input_staging, g_output))
            {
                std::printf("ERR invoke image %d\n", i);
                return;
            }
            const int64_t elapsed_us = esp_timer_get_time() - start_us;
            if (i > 0)
            {
                run_total_us += static_cast<double>(elapsed_us);
                ++counted;
            }
            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10(g_output);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            g_output, sizeof(float) * kOutputClasses);
                if (predicted == sample.label)
                    ++correct;
            }
        }
        run_averages_us[static_cast<size_t>(run)] = run_total_us / static_cast<double>(counted);
    }

    std::printf("\n  per-digit results (final run):\n");
    for (int i = 0; i < kImageCount; ++i)
    {
        const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];
        const int predicted = final_predictions[static_cast<size_t>(i)];
        const float* out = &final_outputs[static_cast<size_t>(i) * kOutputClasses];
        const int ok = predicted == sample.label ? 1 : 0;
        PrintDigitSummary(i, sample.label, predicted, out[predicted], out, ok);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
        mean_us += run_averages_us[static_cast<size_t>(i)];
    mean_us /= static_cast<double>(kRuns);

    std::printf("  accuracy:    %d/%d on final run\n", correct, kImageCount);
    std::printf("\nnetkit MNIST cnn_f32 float32 benchmark summary (reference)\n");
    std::printf("  method:      %d runs x 10 images, discard first invoke each run\n", kRuns);
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_whole = mean_us_i / 1000ul;
    const unsigned long mean_ms_frac = mean_us_i % 1000ul;
    std::printf("  mean:   %lu us (%lu.%03lu ms)\n", mean_us_i, mean_ms_whole, mean_ms_frac);
    std::printf("BENCHMARK_SUMMARY runtime=netkit model=cnn_f32 backend=reference-f32 mean_us=%lu runs=%d\n",
                mean_us_i, kRuns);
    std::printf("\nDONE\n");
}
