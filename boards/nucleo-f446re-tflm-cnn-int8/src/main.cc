// NUCLEO-F446RE TFLM MNIST CNN int8 invoke benchmark — same 10 images as benchmark/netkit.
// Full int8 graph (int8 input/output). CMSIS-NN optimized kernels via TFLM microlite build.
// Test inputs are prequantized int8 (export_int8_test_images.py) — no float conversion.

#include "dwt_time.h"
#include "generated/mnist_cnn_int8_model_data.h"
#include "generated/mnist_cnn_int8_test_images.h"
#include "stm32f446xx.h"
#include "uart.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/reshape.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <cstring>

namespace {

constexpr int kRuns = 10;
constexpr int kTensorArenaSize = 116 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

using MnistCnnOpResolver = tflite::MicroMutableOpResolver<10>;

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

void CopyInputInt8(TfLiteTensor* input, const int8_t* pixels)
{
    std::memcpy(input->data.int8, pixels, kMnistCnnInt8BenchmarkInputSize);
}

}  // namespace

extern "C" void uart_debug_log(const char* message);

extern "C" int main(void)
{
    uart_init();
    dwt_time_init();
    RegisterDebugLogCallback(uart_debug_log);

    tflite::InitializeTarget();

    MnistCnnOpResolver op_resolver;
    if (op_resolver.AddConv2D() != kTfLiteOk || op_resolver.AddMaxPool2D() != kTfLiteOk ||
        op_resolver.AddFullyConnected() != kTfLiteOk || op_resolver.AddSoftmax() != kTfLiteOk ||
        op_resolver.AddReshape() != kTfLiteOk)
    {
        uart_write("ERR op resolver\r\n");
        for (;;)
        {
        }
    }

    const tflite::Model* model = tflite::GetModel(g_mnist_cnn_int8_model_data);
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

    uart_write("\r\nTFLM NUCLEO-F446RE MNIST CNN int8 benchmark\r\n");
    uart_printf("  backend:     tflm cmsis-nn int8 (MCU CM4)\r\n");
    uart_printf("  model bytes: %u\r\n", g_mnist_cnn_int8_model_data_size);
    uart_write("  input type:  int8 (prequantized test vectors)\r\n");
    uart_write("  output type: int8 (argmax on logits)\r\n");
    uart_printf("  images:      %d per run\r\n", kMnistCnnInt8BenchmarkImageCount);
    uart_printf("  runs:        %d (discard first invoke each run)\r\n", kRuns);
    uart_printf("  arena bytes: %d\r\n", kTensorArenaSize);
    uart_printf("  sysclk:      %lu Hz\r\n", static_cast<unsigned long>(SystemCoreClock));

    {
        const MnistCnnInt8BenchmarkSample& probe = kMnistCnnInt8BenchmarkImages[0];
        CopyInputInt8(input, probe.pixels);
        if (interpreter.Invoke() != kTfLiteOk)
        {
            uart_write("ERR probe invoke\r\n");
            for (;;)
            {
            }
        }
        const int predicted = ArgMax10Int8(output->data.int8);
        uart_printf("  probe:       label=%d pred=%d i8[0..3]=%d,%d,%d,%d\r\n",
                    probe.label,
                    predicted,
                    static_cast<int>(output->data.int8[0]),
                    static_cast<int>(output->data.int8[1]),
                    static_cast<int>(output->data.int8[2]),
                    static_cast<int>(output->data.int8[3]));
    }

    std::array<double, kRuns> run_averages_us{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;

        for (int i = 0; i < kMnistCnnInt8BenchmarkImageCount; ++i)
        {
            const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];

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

    uart_printf("  accuracy:    %d/%d on final run\r\n", correct, kMnistCnnInt8BenchmarkImageCount);
    uart_write("\r\nTFLM MNIST cnn int8 benchmark summary (cmsis-nn)\r\n");
    uart_printf("  method:      %d runs x 10 images, discard first invoke each run\r\n", kRuns);
    uart_write("  per-run avg: avg of images 1-9 (us)\r\n\r\n");
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_i = mean_us_i / 1000UL;
    uart_printf("  mean:   %8lu us (%6lu ms)\r\n", mean_us_i, mean_ms_i);
    uart_printf(
        "BENCHMARK_SUMMARY runtime=tflm model=cnn_int8 backend=cmsis-nn mean_us=%lu runs=%d\r\n",
        mean_us_i,
        kRuns);
    uart_write("\r\nDONE\r\n");

    for (;;)
    {
    }
}
