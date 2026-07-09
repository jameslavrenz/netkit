// TFLM MobileNetV4-small float32 invoke-time benchmark (host/CPU).
//
// Pairs with benchmark/netkit/src/mobilenetv4_main.cc: same architecture
// (MobileNetV4-Conv-Small, input 56x56x3, 10 classes), same input pipeline
// (10 MNIST images upsampled 28x28x1 -> 56x56x3), same methodology (loop over
// the images kLoops times, time every Invoke(), report cold first invoke plus
// warm median/min/mean/max/stddev). Fixture weights -> accuracy not meaningful.

#include "generated/mnist_cnn_test_images.h"
#include "generated/mobilenetv4_small_model_data.h"

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

constexpr int kLoops = 30;  // passes over the 10 images -> 300 invokes total
constexpr int kInH = 56;
constexpr int kInW = 56;
constexpr int kInC = 3;
constexpr int kInputSize = kInH * kInW * kInC;
// MobileNetV4-small float32 activations are small (peak ~100 KB); 4 MB is ample.
constexpr int kTensorArenaSize = 4 * 1024 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

using Mnv4OpResolver = tflite::MicroMutableOpResolver<8>;

int ArgMax10(const float* values) {
  int best = 0;
  float max_val = values[0];
  for (int i = 1; i < 10; ++i) {
    if (values[i] > max_val) {
      max_val = values[i];
      best = i;
    }
  }
  return best;
}

// 28x28x1 MNIST -> 56x56x3 NHWC via 2x nearest-neighbour + channel replication.
// Matches benchmark/netkit/src/mobilenetv4_main.cc::UpsampleMnistTo56x56x3.
void UpsampleMnistTo56x56x3(const float* src28, float* dst) {
  for (int y = 0; y < kInH; ++y) {
    for (int x = 0; x < kInW; ++x) {
      const float v = src28[(y / 2) * 28 + (x / 2)];
      float* px = dst + (y * kInW + x) * kInC;
      px[0] = v;
      px[1] = v;
      px[2] = v;
    }
  }
}

TfLiteStatus RunBenchmark() {
  Mnv4OpResolver op_resolver;
  if (op_resolver.AddConv2D() != kTfLiteOk ||
      op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      op_resolver.AddAdd() != kTfLiteOk ||
      op_resolver.AddAveragePool2D() != kTfLiteOk ||
      op_resolver.AddFullyConnected() != kTfLiteOk ||
      op_resolver.AddReshape() != kTfLiteOk) {
    MicroPrintf("Failed to register MobileNetV4 ops");
    return kTfLiteError;
  }

  const tflite::Model* model = tflite::GetModel(g_mobilenetv4_small_model_data);
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
  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
    MicroPrintf("Expected float32 tensors");
    return kTfLiteError;
  }
  if (input->bytes < kInputSize * sizeof(float)) {
    MicroPrintf("Input tensor too small: %d bytes", (int)input->bytes);
    return kTfLiteError;
  }

  const int num_images = kMnistCnnBenchmarkImageCount;
  std::vector<std::vector<float>> inputs(num_images);
  for (int i = 0; i < num_images; ++i) {
    inputs[i].resize(kInputSize);
    UpsampleMnistTo56x56x3(kMnistCnnBenchmarkImages[i].pixels, inputs[i].data());
  }

  MicroPrintf("TFLM MobileNetV4-small benchmark");
  MicroPrintf("  backend:     reference (TFLM builtin float kernels)");
  MicroPrintf("  model bytes: %u", g_mobilenetv4_small_model_data_size);
  MicroPrintf("  input:       %dx%dx%d  outputs: 10", kInH, kInW, kInC);
  MicroPrintf("  method:      %d images x %d loops = %d invokes (all timed)",
              num_images, kLoops, num_images * kLoops);
  MicroPrintf("  arena bytes: %d", kTensorArenaSize);

  std::vector<double> samples;
  samples.reserve(num_images * kLoops);

  int last_pred = -1;
  for (int loop = 0; loop < kLoops; ++loop) {
    for (int i = 0; i < num_images; ++i) {
      std::memcpy(input->data.f, inputs[i].data(), kInputSize * sizeof(float));

      const auto start = std::chrono::steady_clock::now();
      const TfLiteStatus invoke_status = interpreter.Invoke();
      const auto end = std::chrono::steady_clock::now();
      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed on loop %d image %d", loop, i);
        return kTfLiteError;
      }
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
      last_pred = ArgMax10(output->data.f);
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

  // Plain printf for the numeric summary (MicroPrintf lacks %f on host).
  std::printf("\nTFLM MobileNetV4-small benchmark summary (reference)\n");
  std::printf("  cold invoke:      %9.3f us (%7.3f ms)\n", cold_us, cold_us / 1000.0);
  std::printf("  warm median:      %9.3f us (%7.3f ms)  <- primary metric\n",
              warm_median, warm_median / 1000.0);
  std::printf("  warm min:         %9.3f us (%7.3f ms)\n", warm_min, warm_min / 1000.0);
  std::printf("  warm mean:        %9.3f us (%7.3f ms)  over %zu invokes\n",
              warm_mean, warm_mean / 1000.0, warm_n);
  std::printf("  warm max:         %9.3f us\n", warm_max);
  std::printf("  warm stddev:      %9.3f us\n", warm_std);
  std::printf(
      "BENCHMARK_SUMMARY runtime=tflm model=mobilenetv4_small dtype=float32 backend=reference "
      "warm_median_us=%.3f warm_min_us=%.3f warm_mean_us=%.3f cold_us=%.3f "
      "invokes=%zu\n",
      warm_median, warm_min, warm_mean, cold_us, samples.size());

  return kTfLiteOk;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  tflite::InitializeTarget();
  const TfLiteStatus status = RunBenchmark();
  return status == kTfLiteOk ? 0 : 1;
}
