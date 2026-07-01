#pragma once
#include "arena.hpp"
#include "tensor.hpp"
#include <initializer_list>
#include <iostream>
#include <span>

namespace TensorFactory
{
    void Print(const Tensor& t);
    void PrintLabeled(const char* label, const Tensor& t);
    Tensor Create2D(Arena& arena, uint32_t rows, uint32_t cols);
    Tensor CreateND(Arena& arena, uint32_t rank, std::span<const uint32_t> shape);
    Tensor View2D(float* data, uint32_t rows, uint32_t cols);
    Tensor ViewND(float* data, uint32_t rank, std::span<const uint32_t> shape);
    void Fill(Tensor& t, std::initializer_list<float> values);
}
