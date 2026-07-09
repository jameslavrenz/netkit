// NUCLEO-F446RE TFLM MNIST MLP int8 invoke benchmark — same 10 images as benchmark/netkit.

#include "dwt_time.h"
#include "generated/mnist_mlp_int8_model_data.h"
#include "generated/mnist_mlp_int8_test_images.h"
#include "stm32f446xx.h"
#include "uart.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <cstring>

namespace {

#if defined(CMSIS_NN) || defined(CMSISNN)
#define TFLM_MLP_INT8_BACKEND_LABEL "cmsis-nn"
#else
#define TFLM_MLP_INT8_BACKEND_LABEL "reference"
#endif

constexpr int kRuns = 10;
constexpr int kOutputClasses = 10;
constexpr int kTensorArenaSize = 48 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

using MnistMlpOpResolver = tflite::MicroMutableOpResolver<4>;

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

void PrintDigitSummary(int image,
                       int label,
                       int predicted,
                       int pred_i8,
                       const int8_t* out_i8,
                       int ok)
{
    uart_printf(
        "DIGIT_SUMMARY runtime=tflm model=mlp_int8 image=%d label=%d pred=%d pred_i8=%d ok=%d out_i8=",
        image,
        label,
        predicted,
        pred_i8,
        ok);
    PrintOutI8Uart(out_i8);
    uart_write("\r\n");
}

void CopyInputInt8(TfLiteTensor* input, const int8_t* pixels)
{
    std::memcpy(input->data.int8, pixels, kMnistMlpInt8BenchmarkInputSize);
}

}  // namespace

extern "C" void uart_debug_log(const char* message);

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();
    RegisterDebugLogCallback(uart_debug_log);

    tflite::InitializeTarget();

    MnistMlpOpResolver op_resolver;
    if (op_resolver.AddFullyConnected() != kTfLiteOk || op_resolver.AddSoftmax() != kTfLiteOk)
    {
        uart_write("ERR op resolver\r\n");
        for (;;)
        {
        }
    }

    const tflite::Model* model = tflite::GetModel(g_mnist_mlp_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        uart_write("ERR schema version\r\n");
        for (;;)
        {
        }
    }

    tflite::MicroInterpreter interpreter(model, op_resolver, g_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk)
    {
        uart_write("ERR AllocateTensors\r\n");
        for (;;)
        {
        }
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input == nullptr || output == nullptr || input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8)
    {
        uart_write("ERR tensors (expected int8 I/O)\r\n");
        for (;;)
        {
        }
    }

    uart_write("\r\nTFLM NUCLEO-F446RE MNIST MLP int8 benchmark\r\n");
    uart_printf("  backend:     tflm %s int8 (MCU CM4)\r\n", TFLM_MLP_INT8_BACKEND_LABEL);
    uart_printf("  model bytes: %u\r\n", g_mnist_mlp_int8_model_data_size);
    uart_write("  input type:  int8 (prequantized test vectors)\r\n");
    uart_write("  output type: int8 softmax (int8 only on device)\r\n");
    uart_printf("  images:      %d per run\r\n", kMnistMlpInt8BenchmarkImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  arena bytes: %d\r\n", kTensorArenaSize);
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    {
        const MnistMlpInt8BenchmarkSample& probe = kMnistMlpInt8BenchmarkImages[0];
        CopyInputInt8(input, probe.pixels);
        if (interpreter.Invoke() != kTfLiteOk)
        {
            uart_write("ERR probe invoke\r\n");
            for (;;)
            {
            }
        }
        const int predicted = ArgMax10Int8(output->data.int8);
        uart_printf("  probe:       label=%d pred=%d pred_i8=%d out_i8=",
                    probe.label,
                    predicted,
                    static_cast<int>(output->data.int8[predicted]));
        PrintOutI8Uart(output->data.int8);
        uart_write("\r\n");
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kMnistMlpInt8BenchmarkImageCount> final_predictions{};
    std::array<int8_t, kMnistMlpInt8BenchmarkImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistMlpInt8BenchmarkImageCount; ++i)
        {
            const MnistMlpInt8BenchmarkSample& sample = kMnistMlpInt8BenchmarkImages[i];

            CopyInputInt8(input, sample.pixels);

            const uint32_t start_cycles = dwt_cycles();
            const TfLiteStatus invoke_status = interpreter.Invoke();
            const uint32_t elapsed_cycles = dwt_cycles() - start_cycles;

            if (invoke_status != kTfLiteOk)
            {
                uart_printf("ERR invoke run %d image %d\r\n", run + 1, i);
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
                const int predicted = ArgMax10Int8(output->data.int8);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            output->data.int8,
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
    for (int i = 0; i < kMnistMlpInt8BenchmarkImageCount; ++i)
    {
        const MnistMlpInt8BenchmarkSample& sample = kMnistMlpInt8BenchmarkImages[i];
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

    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kMnistMlpInt8BenchmarkImageCount);
    uart_printf("\r\nTFLM MNIST mlp int8 benchmark summary (%s)\r\n", TFLM_MLP_INT8_BACKEND_LABEL);
    uart_printf("  method:      %d runs x 10 images, discard first invoke each run\r\n", kRuns);
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_i = mean_us_i / 1000UL;
    uart_printf("  mean:   %8lu us (%6lu ms)\r\n", mean_us_i, mean_ms_i);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=tflm model=mlp_int8 backend=%s mean_us=%lu runs=%d\r\n",
        TFLM_MLP_INT8_BACKEND_LABEL,
        mean_us_i,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
