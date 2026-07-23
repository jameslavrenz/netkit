// Seeed XIAO ESP32C3 — TFLM MNIST DS-CNN int8 (esp-tflite-micro + ESP-NN).
// Same 10×10 methodology as NUCLEO / netkit XIAO peers: discard image 0 each run.

#include "esp_timer.h"
#include "generated/mnist_cnn_dw_int8_model_data.h"
#include "generated/cnn_dw/mnist_cnn_int8_test_images.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace {

#if defined(ESP_NN)
#define TFLM_CNN_DW_INT8_BACKEND_LABEL "esp-nn"
#else
#define TFLM_CNN_DW_INT8_BACKEND_LABEL "reference"
#endif

constexpr int kRuns = 10;
constexpr int kOutputClasses = 10;
// ESP32-C3 SRAM is tight with IDF; 96 KiB matches published NUCLEO A/B arena budget.
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

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

void PrintOutI8(const int8_t* values)
{
    for (int i = 0; i < kOutputClasses; ++i)
    {
        std::printf("%s%d", i ? "," : "", static_cast<int>(values[i]));
    }
}

void PrintDigitSummary(int image,
                       int label,
                       int predicted,
                       int pred_i8,
                       const int8_t* out_i8,
                       int ok)
{
    std::printf(
        "DIGIT_SUMMARY runtime=tflm model=cnn_dw_int8 image=%d label=%d pred=%d pred_i8=%d ok=%d out_i8=",
        image,
        label,
        predicted,
        pred_i8,
        ok);
    PrintOutI8(out_i8);
    std::printf("\n");
}

void CopyInputInt8(TfLiteTensor* input, const int8_t* pixels)
{
    std::memcpy(input->data.int8, pixels, static_cast<size_t>(kMnistCnnInt8BenchmarkInputSize));
}

}  // namespace

extern "C" void app_main(void)
{
    tflite::InitializeTarget();

    MnistCnnOpResolver op_resolver;
    if (op_resolver.AddConv2D() != kTfLiteOk ||
        op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        op_resolver.AddMaxPool2D() != kTfLiteOk ||
        op_resolver.AddFullyConnected() != kTfLiteOk || op_resolver.AddSoftmax() != kTfLiteOk ||
        op_resolver.AddReshape() != kTfLiteOk)
    {
        std::printf("ERR op resolver\n");
        return;
    }

    const tflite::Model* model = tflite::GetModel(g_mnist_cnn_dw_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        std::printf("ERR schema version\n");
        return;
    }

    tflite::MicroInterpreter interpreter(model, op_resolver, g_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk)
    {
        std::printf("ERR AllocateTensors (arena=%d)\n", kTensorArenaSize);
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if (input == nullptr || output == nullptr || input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8)
    {
        std::printf("ERR tensors (expected int8 I/O)\n");
        return;
    }

    std::printf("\nTFLM XIAO ESP32C3 MNIST DS-CNN int8 benchmark\n");
    std::printf("  backend:     tflm %s int8 (MCU ESP32C3)\n", TFLM_CNN_DW_INT8_BACKEND_LABEL);
    std::printf("  model bytes: %u\n", g_mnist_cnn_dw_int8_model_data_size);
    std::printf("  input type:  int8 (prequantized test vectors)\n");
    std::printf("  output type: int8\n");
    std::printf("  images:      %d per run\n", kMnistCnnInt8BenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  arena bytes: %d\n", kTensorArenaSize);
    std::printf("  sysclk:      160000000 Hz\n");

    {
        const MnistCnnInt8BenchmarkSample& probe = kMnistCnnInt8BenchmarkImages[0];
        CopyInputInt8(input, probe.pixels);
        if (interpreter.Invoke() != kTfLiteOk)
        {
            std::printf("ERR probe invoke\n");
            return;
        }
        const int predicted = ArgMax10Int8(output->data.int8);
        std::printf("  probe:       label=%d pred=%d pred_i8=%d out_i8=",
                    probe.label,
                    predicted,
                    static_cast<int>(output->data.int8[predicted]));
        PrintOutI8(output->data.int8);
        std::printf("\n");
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
            CopyInputInt8(input, sample.pixels);
            const int64_t start_us = esp_timer_get_time();
            if (interpreter.Invoke() != kTfLiteOk)
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

    std::printf("\n  per-digit results (final run):\n");
    std::printf("    image  label  pred  pred_i8  ok\n");
    for (int i = 0; i < kMnistCnnInt8BenchmarkImageCount; ++i)
    {
        const MnistCnnInt8BenchmarkSample& sample = kMnistCnnInt8BenchmarkImages[i];
        const int predicted = final_predictions[static_cast<size_t>(i)];
        const int8_t* out_i8 = &final_outputs[static_cast<size_t>(i) * kOutputClasses];
        const int pred_i8 = static_cast<int>(out_i8[predicted]);
        const int ok = predicted == sample.label ? 1 : 0;
        std::printf("    %5d  %5d  %4d  %7d  %s\n",
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

    std::printf("  accuracy:    %d/%d on final run\n", correct, kMnistCnnInt8BenchmarkImageCount);
    std::printf("\nTFLM MNIST cnn_dw int8 benchmark summary (%s)\n", TFLM_CNN_DW_INT8_BACKEND_LABEL);
    std::printf("  method:      %d runs x 10 images, discard first invoke each run\n", kRuns);
    std::printf("  per-run avg: avg of images 1-9 (us)\n\n");
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_whole = mean_us_i / 1000ul;
    const unsigned long mean_ms_frac = mean_us_i % 1000ul;
    std::printf("  mean:   %lu us (%lu.%03lu ms)\n", mean_us_i, mean_ms_whole, mean_ms_frac);
    std::printf("BENCHMARK_SUMMARY runtime=tflm model=cnn_dw_int8 backend=%s mean_us=%lu runs=%d\n",
                TFLM_CNN_DW_INT8_BACKEND_LABEL,
                mean_us_i,
                kRuns);
    std::printf("\nDONE\n");
}
