#pragma once

#include <cstdint>

constexpr int kMobilenetV4Int8BenchmarkImageCount = 10;
constexpr int kMobilenetV4Int8BenchmarkInputSize = 9408;
constexpr float kMobilenetV4Int8BenchmarkInputScale = 0.00658859f;
constexpr int kMobilenetV4Int8BenchmarkInputZeroPoint = -6;

struct MobilenetV4Int8BenchmarkSample {
  const char* name;
  int label;
  const int8_t* pixels;
};

extern const MobilenetV4Int8BenchmarkSample kMobilenetV4Int8BenchmarkImages[10];
