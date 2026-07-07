// NUCLEO-F446RE MNIST CNN int8 invoke benchmark — same 10 images as benchmark/netkit.
// Supports interpreter embed (default) or quant lowered embed (NETKIT_LOWERED=1).
// Test inputs are prequantized int8 (export_int8_test_images.py) — no float conversion.
// Dequantized confidence is computed offline — see benchmark/tools/parse_mcu_cnn_int8_log.py.

#include "dwt_time.h"
#include "mnist_cnn_int8_aot.hpp"
#include "mnist_cnn_int8_test_images.h"
#include "netkit_config.h"
#include "quant_trace.hpp"
#include "stm32f446xx.h"
#include "uart.h"

#include <array>
#include <cstring>

namespace aot = netkit::aot::mnist_cnn_int8;

namespace {

constexpr int kRuns = 10;
constexpr int kImageCount = kMnistCnnInt8BenchmarkImageCount;
constexpr int kInputSize = kMnistCnnInt8BenchmarkInputSize;
constexpr int kOutputClasses = 10;
// Interpreter embed path can emit kArenaBytesRecommended=0 when host probe fails; use MCU slack.
constexpr std::size_t kArenaCapacity =
    aot::kArenaBytesRecommended > 0 ? aot::kArenaBytesRecommended : (64u * 1024u);

alignas(std::max_align_t) static unsigned char g_arena_memory[kArenaCapacity];
alignas(std::max_align_t) static int8_t g_output_i8[aot::kOutputElements];

int ArgMax10Int8(const int8_t* values)
{
    int best = 0;
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > values[best])
            best = i;
    }
    return best;
}

void PrintOutI8Uart(const int8_t* values)
{
    for (int i = 0; i < kOutputClasses; ++i)
    {
        uart_printf("%s%d", i ? "," : "", static_cast<int>(values[i]));
    }
}

void PrintDigitSummary(int image,
                       int label,
                       int predicted,
                       int pred_i8,
                       const int8_t* out_i8,
                       int ok)
{
    uart_printf(
        "DIGIT_SUMMARY runtime=netkit model=cnn_int8 image=%d label=%d pred=%d pred_i8=%d ok=%d out_i8=",
        image,
        label,
        predicted,
        pred_i8,
        ok);
    PrintOutI8Uart(out_i8);
    uart_write("\r\n");
}

template <bool Lowered>
void PrintStorageInfo()
{
    if constexpr (Lowered)
        uart_printf("  workspace:   %u bytes\r\n", static_cast<unsigned>(aot::kWorkspaceBytes));
    else
        uart_printf("  nk bytes:    %u\r\n", static_cast<unsigned>(aot::kNkBytes));
}

}  // namespace

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();

    uart_write("\r\nnetkit NUCLEO-F446RE MNIST CNN int8 benchmark\r\n");
    uart_printf("  backend:     cmsis-nn int8 (MCU CM4%s)\r\n",
                aot::kQuantLowered ? ", quant lowered AOT" : ", .nk loader");
    uart_printf("  weights:     %s\r\n",
                aot::kQuantLowered ? "flash (static .rodata)"
                                   : (NETKIT_WEIGHTS_IN_RAM ? "ram (arena copy at load)"
                                                            : "flash (embedded .nk blob)"));
    uart_write("  dtype:       int8 end-to-end (weights, activations, inputs, softmax)\r\n");
    uart_printf("  images:      %d per run\r\n", kImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  arena bytes: %u\r\n", static_cast<unsigned>(kArenaCapacity));
    PrintStorageInfo<aot::kQuantLowered>();
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    Arena arena;
    if (!aot::InitArena(arena, g_arena_memory, sizeof(g_arena_memory)))
    {
        uart_write("ERR arena init\r\n");
        for (;;)
        {
        }
    }

    QuantTrace::Reset();

    aot::Model model;
    if (!model.load(arena))
    {
        uart_write("ERR model load\r\n");
        for (;;)
        {
        }
    }
    uart_write("  model:       loaded\r\n");

    {
        const MnistCnnInt8BenchmarkSample& probe = kMnistCnnInt8BenchmarkImages[0];
        if (!model.forwardInt8(arena, probe.pixels, g_output_i8))
        {
            uart_write("ERR probe forward\r\n");
            for (;;)
            {
            }
        }
        const int predicted = ArgMax10Int8(g_output_i8);
        uart_printf("  probe:       label=%d pred=%d pred_i8=%d out_i8=",
                    probe.label,
                    predicted,
                    static_cast<int>(g_output_i8[predicted]));
        PrintOutI8Uart(g_output_i8);
        uart_write("\r\n");
        QuantTrace::PrintSummaryUart();
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kImageCount> final_predictions{};
    std::array<int8_t, kImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kImageCount; ++i)
        {
            const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];

            const uint32_t start_cycles = dwt_cycles();
            if (!model.forwardInt8(arena, sample.pixels, g_output_i8))
            {
                uart_printf("ERR invoke image %d\r\n", i);
                for (;;)
                {
                }
            }
            const uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

            const double elapsed_us = dwt_cycles_to_us(elapsed_cycles);
            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10Int8(g_output_i8);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            g_output_i8,
                            static_cast<size_t>(kOutputClasses));
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
    }

    uart_write("\r\n  per-digit results (final run, int8 only — dequant in Python):\r\n");
    uart_write("    image  label  pred  pred_i8  ok\r\n");
    for (int i = 0; i < kImageCount; ++i)
    {
        const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
        const int predicted = final_predictions[static_cast<size_t>(i)];
        const int8_t* out_i8 = &final_outputs[static_cast<size_t>(i) * kOutputClasses];
        const int pred_i8 = static_cast<int>(out_i8[predicted]);
        const int ok = predicted == sample.label ? 1 : 0;
        uart_printf("    %5d  %5d  %4d  %7d  %s\r\n",
                    i,
                    sample.label,
                    predicted,
                    pred_i8,
                    ok ? "yes" : "no");
        PrintDigitSummary(i, sample.label, predicted, pred_i8, out_i8, ok);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
    {
        mean_us += run_averages_us[static_cast<size_t>(i)];
    }
    mean_us /= static_cast<double>(kRuns);

    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kImageCount);
    uart_write("\r\nnetkit MNIST cnn int8 benchmark summary\r\n");
    uart_printf("  method:      %d runs x 10 images, discard first invoke each run\r\n", kRuns);
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    uart_printf("  mean:   %8.3f us (%6.3f ms)\r\n", mean_us, mean_us / 1000.0);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=netkit model=cnn_int8 backend=cmsis-nn-int8 mean_us=%.3f runs=%d\r\n",
        mean_us,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
