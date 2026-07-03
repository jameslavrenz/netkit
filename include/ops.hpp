#pragma once
#include <cstdint>
#include <cmath>
#include "netkit_log.hpp"
#include "tensor.hpp"

namespace Ops
{
    // Validation functions
    bool IsElementwiseValid(const Tensor& A, const Tensor& B);
    bool CheckSameShape2D(const Tensor& A, const Tensor& B, const Tensor& C);
    bool CheckSameShapeND(const Tensor& A, const Tensor& B, const Tensor& C);
    bool IsMatMulValid(const Tensor& A, const Tensor& B, const Tensor& C);
    bool IsElementwiseValidND(const Tensor& A, const Tensor& B, const Tensor& C);
    bool IsUnaryOpValid(const Tensor& A, const Tensor& C);

    // Arithmetic operations
    void Mul(const Tensor& A, const Tensor& B, Tensor& C);
    void MulScalar(const Tensor& A, float scalar, Tensor& C);
    void MatAdd(const Tensor& A, const Tensor& B, Tensor& C);
    void MatAddND(const Tensor& A, const Tensor& B, Tensor& C);
    void MatMul(const Tensor& A, const Tensor& B, Tensor& C);
    void MulND(const Tensor& A, const Tensor& B, Tensor& C);

    // Activation functions
    void ReLU(const Tensor& A, Tensor& C);
    void Sigmoid(const Tensor& A, Tensor& C);
    void Tanh(const Tensor& A, Tensor& C);
    void LeakyReLU(const Tensor& A, Tensor& C, float alpha = 0.01f);
    void ReLU6(const Tensor& A, Tensor& C);
    void Softmax(const Tensor& A, Tensor& C);
}
