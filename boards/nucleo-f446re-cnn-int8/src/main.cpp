// NUCLEO-F446RE MNIST CNN int8 invoke benchmark — same 10 images as benchmark/netkit.
// Quant lowered AOT: static int8 weights + CmsisQuantPlan call chain (no .nk loader).
// Test inputs are prequantized int8 (export_int8_test_images.py) — no float conversion.

#include "dwt_time.h"
#include "mnist_cnn_int8_aot.hpp"
#include "mnist_cnn_int8_test_images.h"
#include "netkit_config.h"
#include "quant_output.hpp"
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

alignas(std::max_align_t) static unsigned char g_arena_memory[aot::kArenaBytesRecommended];
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
    uart_printf("  arena bytes: %u\r\n", static_cast<unsigned>(aot::kArenaBytesRecommended));
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
        uart_printf("  probe:       label=%d pred=%d i8[0..3]=%d,%d,%d,%d\r\n",
                    probe.label,
                    predicted,
                    static_cast<int>(g_output_i8[0]),
                    static_cast<int>(g_output_i8[1]),
                    static_cast<int>(g_output_i8[2]),
                    static_cast<int>(g_output_i8[3]));
        QuantTrace::PrintSummaryUart();
    }

    std::array<double, kRuns> run_averages_us{};
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
                if (predicted == sample.label)
                {
                    ++correct;
                }
            }
        }

        run_averages_us[static_cast<size_t>(run)] =
            run_total_us / static_cast<double>(counted);
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
