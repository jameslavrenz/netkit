#pragma once
#include <array>
#include <cstdint>

enum class DataType : uint8_t
{
    Float32,  // IEEE-754 single precision — all inference tensors and weights use float32
    Int8,
    UInt8,
    Int16
};

constexpr uint32_t kMaxTensorRank = 4;

struct Tensor
{
    void* data;
    DataType type;

    uint32_t rank;
    std::array<uint32_t, kMaxTensorRank> shape{};
    std::array<uint32_t, kMaxTensorRank> stride{};
    uint32_t num_elements;
    uint32_t bytes;
};
