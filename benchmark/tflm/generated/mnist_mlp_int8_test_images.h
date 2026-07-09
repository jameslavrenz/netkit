#pragma once

#include <cstdint>

constexpr int kMnistMlpInt8BenchmarkImageCount = 10;
constexpr int kMnistMlpInt8BenchmarkInputSize = 784;
constexpr float kMnistMlpInt8BenchmarkInputScale = 0.00392157f;
constexpr int kMnistMlpInt8BenchmarkInputZeroPoint = -128;

struct MnistMlpInt8BenchmarkSample {
  const char* name;
  int label;
  const int8_t* pixels;
};

extern const MnistMlpInt8BenchmarkSample kMnistMlpInt8BenchmarkImages[10];
