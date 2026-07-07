#pragma once

#include <cstdint>

constexpr int kMnistCnnInt8BenchmarkImageCount = 10;
constexpr int kMnistCnnInt8BenchmarkInputSize = 784;
constexpr float kMnistCnnInt8BenchmarkInputScale = 0.00392157f;
constexpr int kMnistCnnInt8BenchmarkInputZeroPoint = -128;

struct MnistCnnInt8BenchmarkSample {
  const char* name;
  int label;
  const int8_t* pixels;
};

extern const MnistCnnInt8BenchmarkSample kMnistCnnInt8BenchmarkImages[10];
