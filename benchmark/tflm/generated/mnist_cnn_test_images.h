#pragma once

#include <cstdint>

constexpr int kMnistCnnBenchmarkImageCount = 10;
constexpr int kMnistCnnBenchmarkInputSize = 784;

struct MnistCnnBenchmarkSample {
  const char* name;
  int label;
  const float* pixels;
};

extern const MnistCnnBenchmarkSample kMnistCnnBenchmarkImages[10];

