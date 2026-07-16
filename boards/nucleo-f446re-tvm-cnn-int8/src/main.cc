// NUCLEO-F446RE microTVM AOT MNIST CNN int8 benchmark — same 10 images as netkit/TFLM.

#include "dwt_time.h"
#include "generated/mnist_cnn_int8_test_images.h"
#include "stm32f446xx.h"
#include "uart.h"

#include "tvmgen_mnist_cnn_int8.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kRuns = 10;
constexpr int kOutputClasses = 10;

int ArgMax10Int8(const int8_t* values)
{
    int best = 0;
    for (int i = 1; i < 10; ++i)
    {
        if (values[i] > values[best])
        {
            best = i;
        }
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

void PrintDigitSummary(int image, int label, int predicted, int pred_i8, const int8_t* out_i8, int ok)
{
    uart_printf(
        "DIGIT_SUMMARY runtime=tvm model=cnn_int8 image=%d label=%d pred=%d pred_i8=%d ok=%d out_i8=",
        image,
        label,
        predicted,
        pred_i8,
        ok);
    PrintOutI8Uart(out_i8);
    uart_write("\r\n");
}

alignas(16) static int8_t g_input[kMnistCnnInt8BenchmarkInputSize];
alignas(16) static int8_t g_output[kOutputClasses];

// IO field names come from tools/gen_tvm_io_bind.py after compile-model.
#include "tvm_io_bind.h"

int InvokeOnce()
{
    struct tvmgen_mnist_cnn_int8_inputs inputs = {};
    struct tvmgen_mnist_cnn_int8_outputs outputs = {};
    NETKIT_TVM_BIND_INPUT(inputs, g_input);
    NETKIT_TVM_BIND_OUTPUT(outputs, g_output);
    return tvmgen_mnist_cnn_int8_run(&inputs, &outputs);
}

}  // namespace

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();

    uart_write("\r\nTVM NUCLEO-F446RE MNIST CNN int8 benchmark\r\n");
#ifndef TVM_MCU_BACKEND_LABEL
#define TVM_MCU_BACKEND_LABEL "cmsis-nn"
#endif
    uart_printf("  backend:     microTVM AOT %s int8 (MCU CM4)\r\n", TVM_MCU_BACKEND_LABEL);
    uart_write("  input type:  int8 (prequantized test vectors)\r\n");
    uart_write("  output type: int8 logits/softmax\r\n");
    uart_printf("  images:      %d per run\r\n", kMnistCnnInt8BenchmarkImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    {
        const MnistCnnInt8BenchmarkSample& probe = kMnistCnnInt8BenchmarkImages[0];
        std::memcpy(g_input, probe.pixels, kMnistCnnInt8BenchmarkInputSize);
        if (InvokeOnce() != 0)
        {
            uart_write("ERR probe invoke\r\n");
            for (;;)
            {
            }
        }
        const int predicted = ArgMax10Int8(g_output);
        uart_printf("  probe:       label=%d pred=%d pred_i8=%d out_i8=",
                    probe.label,
                    predicted,
                    static_cast<int>(g_output[predicted]));
        PrintOutI8Uart(g_output);
        uart_write("\r\n");
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kMnistCnnInt8BenchmarkImageCount> final_predictions{};
    std::array<int8_t, kMnistCnnInt8BenchmarkImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistCnnInt8BenchmarkImageCount; ++i)
        {
            const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
            std::memcpy(g_input, sample.pixels, kMnistCnnInt8BenchmarkInputSize);

            const uint32_t start_cycles = dwt_cycles();
            const int status = InvokeOnce();
            const uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

            if (status != 0)
            {
                uart_printf("ERR invoke run %d image %d status=%d\r\n", run + 1, i, status);
                for (;;)
                {
                }
            }

            const double elapsed_us = dwt_cycles_to_us(elapsed_cycles);
            if (i > 0)
            {
                run_total_us += elapsed_us;
                ++counted;
            }

            if (run == kRuns - 1)
            {
                const int predicted = ArgMax10Int8(g_output);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            g_output,
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

    uart_write("\r\n  per-digit results (final run):\r\n");
    for (int i = 0; i < kMnistCnnInt8BenchmarkImageCount; ++i)
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

    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_i = mean_us_i / 1000UL;
    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kMnistCnnInt8BenchmarkImageCount);
    uart_printf("\r\nTVM MNIST cnn int8 benchmark summary (microTVM AOT %s)\r\n",
                TVM_MCU_BACKEND_LABEL);
    uart_printf("  method:      %d runs x 10 images, discard first invoke each run\r\n", kRuns);
    uart_printf("  mean:   %8lu us (%6lu ms)\r\n", mean_us_i, mean_ms_i);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=tvm model=cnn_int8 backend=%s mean_us=%lu runs=%d\r\n",
        TVM_MCU_BACKEND_LABEL,
        mean_us_i,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
