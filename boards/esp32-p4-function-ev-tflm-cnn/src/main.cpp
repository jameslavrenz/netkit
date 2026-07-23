// Seeed ESP32-P4-Function-EV — TFLM MNIST CNN float32.
// Same 10×10 methodology as int8 peers. ESP-NN unused (no float kernels).

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "mnist_cnn_model_data.h"
#include "mnist_cnn_test_images.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kRuns = 10;
constexpr int kOutputClasses = 10;
constexpr int kTensorArenaSize = 256 * 1024;

static uint8_t* g_tensor_arena = nullptr;

using MnistCnnOpResolver = tflite::MicroMutableOpResolver<10>;

int ArgMax10(const float* values)
{
    int best = 0;
    for (int i = 1; i < 10; ++i)
        if (values[i] > values[best]) best = i;
    return best;
}

void PrintOutF32(const float* values)
{
    for (int i = 0; i < kOutputClasses; ++i)
        std::printf("%s%.4f", i ? "," : "", static_cast<double>(values[i]));
}

void PrintDigitSummary(int image, int label, int predicted, float pred_f, const float* out, int ok)
{
    std::printf(
        "DIGIT_SUMMARY runtime=tflm model=cnn_f32 image=%d label=%d pred=%d pred_f=%.4f ok=%d out_f=",
        image, label, predicted, static_cast<double>(pred_f), ok);
    PrintOutF32(out);
    std::printf("\n");
}

}  // namespace

extern "C" void app_main(void)
{
    tflite::InitializeTarget();

    g_tensor_arena = static_cast<uint8_t*>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (g_tensor_arena == nullptr)
    {
        std::printf("ERR arena alloc\n");
        return;
    }

    MnistCnnOpResolver op_resolver;
    if (op_resolver.AddConv2D() != kTfLiteOk || op_resolver.AddMaxPool2D() != kTfLiteOk ||
        op_resolver.AddFullyConnected() != kTfLiteOk || op_resolver.AddSoftmax() != kTfLiteOk ||
        op_resolver.AddReshape() != kTfLiteOk || op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        op_resolver.AddAveragePool2D() != kTfLiteOk)
    {
        std::printf("ERR op resolver\n");
        return;
    }

    const tflite::Model* model = tflite::GetModel(g_mnist_cnn_model_data);
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
    if (input == nullptr || output == nullptr || input->type != kTfLiteFloat32 ||
        output->type != kTfLiteFloat32)
    {
        std::printf("ERR tensors (expected float32 I/O)\n");
        return;
    }

    std::printf("\nTFLM ESP32-P4-Function-EV MNIST CNN float32 benchmark\n");
    std::printf("  backend:     tflm reference float32 (MCU ESP32P4 FPU)\n");
    std::printf("  model bytes: %u\n", g_mnist_cnn_model_data_size);
    std::printf("  arena bytes: %d\n", kTensorArenaSize);
    std::printf("  images:      %d per run\n", kMnistCnnBenchmarkImageCount);
    std::printf("  runs:        %d (discard first invoke each run)\n", kRuns);
    std::printf("  sysclk:      360000000 Hz\n");

    {
        const MnistCnnBenchmarkSample& probe = kMnistCnnBenchmarkImages[0];
        std::memcpy(input->data.f, probe.pixels, sizeof(float) * kMnistCnnBenchmarkInputSize);
        if (interpreter.Invoke() != kTfLiteOk)
        {
            std::printf("ERR probe invoke\n");
            return;
        }
        const int predicted = ArgMax10(output->data.f);
        std::printf("  probe:       label=%d pred=%d pred_f=%.4f\n",
                    probe.label, predicted, static_cast<double>(output->data.f[predicted]));
    }

    std::array<double, kRuns> run_averages_us{};
    std::array<int, kMnistCnnBenchmarkImageCount> final_predictions{};
    std::array<float, kMnistCnnBenchmarkImageCount * kOutputClasses> final_outputs{};
    int correct = 0;

    for (int run = 0; run < kRuns; ++run)
    {
        double run_total_us = 0.0;
        int counted = 0;
        for (int i = 0; i < kMnistCnnBenchmarkImageCount; ++i)
        {
            const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];
            std::memcpy(input->data.f, sample.pixels, sizeof(float) * kMnistCnnBenchmarkInputSize);
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
                const int predicted = ArgMax10(output->data.f);
                final_predictions[static_cast<size_t>(i)] = predicted;
                std::memcpy(&final_outputs[static_cast<size_t>(i) * kOutputClasses],
                            output->data.f, sizeof(float) * kOutputClasses);
                if (predicted == sample.label)
                    ++correct;
            }
        }
        run_averages_us[static_cast<size_t>(run)] = run_total_us / static_cast<double>(counted);
    }

    for (int i = 0; i < kMnistCnnBenchmarkImageCount; ++i)
    {
        const MnistCnnBenchmarkSample& sample = kMnistCnnBenchmarkImages[i];
        const int predicted = final_predictions[static_cast<size_t>(i)];
        const float* out = &final_outputs[static_cast<size_t>(i) * kOutputClasses];
        PrintDigitSummary(i, sample.label, predicted, out[predicted], out,
                          predicted == sample.label ? 1 : 0);
    }

    double mean_us = 0.0;
    for (int i = 0; i < kRuns; ++i)
        mean_us += run_averages_us[static_cast<size_t>(i)];
    mean_us /= static_cast<double>(kRuns);

    std::printf("  accuracy:    %d/%d on final run\n", correct, kMnistCnnBenchmarkImageCount);
    const unsigned long mean_us_i = static_cast<unsigned long>(mean_us + 0.5);
    const unsigned long mean_ms_whole = mean_us_i / 1000ul;
    const unsigned long mean_ms_frac = mean_us_i % 1000ul;
    std::printf("\nTFLM MNIST cnn_f32 float32 benchmark summary (reference)\n");
    std::printf("  mean:   %lu us (%lu.%03lu ms)\n", mean_us_i, mean_ms_whole, mean_ms_frac);
    std::printf("BENCHMARK_SUMMARY runtime=tflm model=cnn_f32 backend=reference-f32 mean_us=%lu runs=%d\n",
                mean_us_i, kRuns);
    std::printf("\nDONE\n");
}
