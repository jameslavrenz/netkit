// TFLM MobileNetV4-Conv-Small int8 ImageNet host benchmark.
//
// Pairs with benchmark/netkit/src/mobilenetv4_imagenet_main.cc on
// models/mobilenetv4_imagenet_int8.nk: same 10 ImageNet samples (prequantized
// 224x224x3), same 5-loop methodology, top-1 accuracy + latency.

#include "generated/imagenet_mnv4_int8_test_images.h"
#include "generated/mobilenetv4_imagenet_int8_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kLoops = 5;
constexpr int kInH = kImagenetMnv4Int8BenchmarkHeight;
constexpr int kInW = kImagenetMnv4Int8BenchmarkWidth;
constexpr int kInC = kImagenetMnv4Int8BenchmarkChannels;
constexpr int kInputSize = kImagenetMnv4Int8BenchmarkInputSize;
constexpr int kNumClasses = kImagenetMnv4Int8BenchmarkNumClasses;
// Large host arena for 224x224 int8 MobileNetV4 activations.
constexpr int kTensorArenaSize = 256 * 1024 * 1024;

using Mnv4Int8OpResolver = tflite::MicroMutableOpResolver<16>;

// Positive output scale ⇒ argmax(int8) == argmax(dequant(int8)); no C++ dequant.
int ArgMaxInt8(const int8_t* values, int count) {
  int best = 0;
  for (int i = 1; i < count; ++i) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

TfLiteStatus RunBenchmark() {
  Mnv4Int8OpResolver op_resolver;
  if (op_resolver.AddConv2D() != kTfLiteOk ||
      op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      op_resolver.AddAdd() != kTfLiteOk ||
      op_resolver.AddAveragePool2D() != kTfLiteOk ||
      op_resolver.AddFullyConnected() != kTfLiteOk ||
      op_resolver.AddReshape() != kTfLiteOk ||
      op_resolver.AddSoftmax() != kTfLiteOk ||
      op_resolver.AddMul() != kTfLiteOk ||
      op_resolver.AddSub() != kTfLiteOk ||
      op_resolver.AddMean() != kTfLiteOk ||
      op_resolver.AddPad() != kTfLiteOk ||
      op_resolver.AddTranspose() != kTfLiteOk ||
      op_resolver.AddRelu() != kTfLiteOk ||
      op_resolver.AddQuantize() != kTfLiteOk ||
      op_resolver.AddDequantize() != kTfLiteOk) {
    MicroPrintf("Failed to register MobileNetV4 ImageNet int8 ops");
    return kTfLiteError;
  }

  const tflite::Model* model =
      tflite::GetModel(g_mobilenetv4_imagenet_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

  std::vector<uint8_t> arena_storage(static_cast<size_t>(kTensorArenaSize));
  uint8_t* tensor_arena = arena_storage.data();

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed (arena too small?)");
    return kTfLiteError;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input == nullptr || output == nullptr) {
    MicroPrintf("Missing input or output tensor");
    return kTfLiteError;
  }
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    MicroPrintf("Expected int8 tensors");
    return kTfLiteError;
  }
  if (input->bytes < kInputSize) {
    MicroPrintf("Input tensor too small: %d bytes", (int)input->bytes);
    return kTfLiteError;
  }
  const int out_count = static_cast<int>(output->bytes);
  if (out_count < kNumClasses) {
    MicroPrintf("Output tensor too small: %d bytes", out_count);
    return kTfLiteError;
  }

  const int num_images = kImagenetMnv4Int8BenchmarkImageCount;

  MicroPrintf("TFLM MobileNetV4 ImageNet int8 benchmark");
  MicroPrintf("  backend:     reference (TFLM builtin int8 kernels)");
  MicroPrintf("  model bytes: %u", g_mobilenetv4_imagenet_int8_model_data_size);
  MicroPrintf("  input:       %dx%dx%d  outputs: %d", kInH, kInW, kInC, out_count);
  MicroPrintf("  method:      %d images x %d loops = %d invokes (all timed)",
              num_images, kLoops, num_images * kLoops);
  MicroPrintf("  arena bytes: %d", kTensorArenaSize);

  std::vector<double> samples;
  samples.reserve(num_images * kLoops);
  int correct = 0;

  for (int loop = 0; loop < kLoops; ++loop) {
    for (int i = 0; i < num_images; ++i) {
      std::memcpy(input->data.int8, kImagenetMnv4Int8BenchmarkImages[i].pixels,
                  kInputSize);

      const auto start = std::chrono::steady_clock::now();
      const TfLiteStatus invoke_status = interpreter.Invoke();
      const auto end = std::chrono::steady_clock::now();
      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed on loop %d image %d", loop, i);
        return kTfLiteError;
      }
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
      const int pred = ArgMaxInt8(output->data.int8, out_count);
      if (loop == 0) {
        if (pred == kImagenetMnv4Int8BenchmarkImages[i].label) ++correct;
        std::printf("  image %d %-28s label=%4d pred=%4d %s\n", i,
                    kImagenetMnv4Int8BenchmarkImages[i].name,
                    kImagenetMnv4Int8BenchmarkImages[i].label, pred,
                    pred == kImagenetMnv4Int8BenchmarkImages[i].label ? "OK"
                                                                      : "MISS");
      }
    }
  }

  const double cold_us = samples.front();
  double first_pass_sum = 0.0;
  for (int i = 0; i < num_images; ++i) first_pass_sum += samples[i];
  const double first_pass_mean = first_pass_sum / static_cast<double>(num_images);

  double warm_sum = 0.0;
  double warm_min = samples[1];
  double warm_max = samples[1];
  const size_t warm_n = samples.size() - 1;
  for (size_t k = 1; k < samples.size(); ++k) {
    warm_sum += samples[k];
    if (samples[k] < warm_min) warm_min = samples[k];
    if (samples[k] > warm_max) warm_max = samples[k];
  }
  const double warm_mean = warm_sum / static_cast<double>(warm_n);
  double var = 0.0;
  for (size_t k = 1; k < samples.size(); ++k) {
    const double d = samples[k] - warm_mean;
    var += d * d;
  }
  const double warm_std = std::sqrt(var / static_cast<double>(warm_n));
  std::vector<double> warm_sorted(samples.begin() + 1, samples.end());
  std::sort(warm_sorted.begin(), warm_sorted.end());
  const double warm_median = warm_sorted[warm_sorted.size() / 2];

  const double top1 = 100.0 * static_cast<double>(correct) / static_cast<double>(num_images);
  std::printf("\nTFLM MobileNetV4 ImageNet int8 summary (reference)\n");
  std::printf("  top-1 accuracy:   %d / %d  (%.1f%%)\n", correct, num_images, top1);
  std::printf("  10-image mean:    %9.3f us (%7.3f ms)  <- primary latency\n",
              first_pass_mean, first_pass_mean / 1000.0);
  std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
  std::printf("  warm median:      %9.3f us (%7.3f ms)\n", warm_median,
              warm_median / 1000.0);
  std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n",
              warm_mean, warm_mean / 1000.0, warm_n);
  std::printf("  warm min/max:     %9.3f / %.3f us\n", warm_min, warm_max);
  std::printf("  warm stddev:      %9.3f us\n", warm_std);
  std::printf(
      "BENCHMARK_SUMMARY runtime=tflm model=mobilenetv4_imagenet dtype=int8 "
      "backend=reference top1_correct=%d top1_total=%d top1_pct=%.1f "
      "ten_image_mean_us=%.3f warm_median_us=%.3f warm_mean_us=%.3f cold_us=%.3f "
      "invokes=%zu\n",
      correct, num_images, top1, first_pass_mean, warm_median, warm_mean, cold_us,
      samples.size());

  return kTfLiteOk;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  tflite::InitializeTarget();
  return RunBenchmark() == kTfLiteOk ? 0 : 1;
}
