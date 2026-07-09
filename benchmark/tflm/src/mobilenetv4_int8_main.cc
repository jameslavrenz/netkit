// TFLM MobileNetV4-small int8 invoke-time benchmark (host/CPU).
//
// Pairs with benchmark/netkit/src/mobilenetv4_main.cc on models/mobilenetv4_small_int8.nk:
// same 10 MNIST-derived inputs (prequantized 56x56x3), same 30-loop methodology.

#include "generated/mobilenetv4_int8_test_images.h"
#include "generated/mobilenetv4_small_int8_model_data.h"

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

constexpr int kLoops = 30;
constexpr int kInH = 56;
constexpr int kInW = 56;
constexpr int kInC = 3;
constexpr int kInputSize = kInH * kInW * kInC;
// Int8 MobileNetV4-small needs a larger arena than float32 on TFLM host.
constexpr int kTensorArenaSize = 16 * 1024 * 1024;
alignas(16) uint8_t* tensor_arena = nullptr;

using Mnv4Int8OpResolver = tflite::MicroMutableOpResolver<8>;

int ArgMax10Int8(const int8_t* values, float scale, int zero_point) {
  int best = 0;
  float max_val = (static_cast<float>(values[0]) - zero_point) * scale;
  for (int i = 1; i < 10; ++i) {
    const float v = (static_cast<float>(values[i]) - zero_point) * scale;
    if (v > max_val) {
      max_val = v;
      best = i;
    }
  }
  return best;
}

TfLiteStatus RunBenchmark() {
  tensor_arena = new uint8_t[kTensorArenaSize];
  if (!tensor_arena) {
    MicroPrintf("Failed to allocate tensor arena");
    return kTfLiteError;
  }

  Mnv4Int8OpResolver op_resolver;
  if (op_resolver.AddConv2D() != kTfLiteOk ||
      op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      op_resolver.AddAdd() != kTfLiteOk ||
      op_resolver.AddAveragePool2D() != kTfLiteOk ||
      op_resolver.AddFullyConnected() != kTfLiteOk ||
      op_resolver.AddReshape() != kTfLiteOk) {
    MicroPrintf("Failed to register MobileNetV4 int8 ops");
    return kTfLiteError;
  }

  const tflite::Model* model = tflite::GetModel(g_mobilenetv4_small_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema %d != supported %d", model->version(),
                TFLITE_SCHEMA_VERSION);
    return kTfLiteError;
  }

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

  const int num_images = kMobilenetV4Int8BenchmarkImageCount;

  MicroPrintf("TFLM MobileNetV4-small int8 benchmark");
  MicroPrintf("  backend:     reference (TFLM builtin int8 kernels)");
  MicroPrintf("  model bytes: %u", g_mobilenetv4_small_int8_model_data_size);
  MicroPrintf("  input:       %dx%dx%d  outputs: 10", kInH, kInW, kInC);
  MicroPrintf("  method:      %d images x %d loops = %d invokes (all timed)",
              num_images, kLoops, num_images * kLoops);
  MicroPrintf("  arena bytes: %d", kTensorArenaSize);

  std::vector<double> samples;
  samples.reserve(num_images * kLoops);

  const float out_scale = output->params.scale;
  const int out_zp = output->params.zero_point;

  int last_pred = -1;
  for (int loop = 0; loop < kLoops; ++loop) {
    for (int i = 0; i < num_images; ++i) {
      std::memcpy(input->data.int8, kMobilenetV4Int8BenchmarkImages[i].pixels,
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
      last_pred = ArgMax10Int8(output->data.int8, out_scale, out_zp);
    }
  }

  const double cold_us = samples.front();
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

  MicroPrintf("  last pred:   class %d (fixture weights; accuracy not meaningful)",
              last_pred);

  std::printf("\nTFLM MobileNetV4-small int8 benchmark summary (reference)\n");
  std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
  std::printf("  warm median:      %9.3f us (%7.3f ms)  <- primary metric\n",
              warm_median, warm_median / 1000.0);
  std::printf("  warm min:         %9.3f us (%7.3f ms)\n", warm_min, warm_min / 1000.0);
  std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n",
              warm_mean, warm_mean / 1000.0, warm_n);
  std::printf("  warm max:         %9.3f us\n", warm_max);
  std::printf("  warm stddev:      %9.3f us\n", warm_std);
  std::printf(
      "BENCHMARK_SUMMARY runtime=tflm model=mobilenetv4_small dtype=int8 backend=reference "
      "warm_median_us=%.3f warm_min_us=%.3f warm_mean_us=%.3f cold_us=%.3f "
      "invokes=%zu\n",
      warm_median, warm_min, warm_mean, cold_us, samples.size());

  return kTfLiteOk;
}

void CleanupArena() {
  delete[] tensor_arena;
  tensor_arena = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  tflite::InitializeTarget();
  const TfLiteStatus status = RunBenchmark();
  CleanupArena();
  return status == kTfLiteOk ? 0 : 1;
}
