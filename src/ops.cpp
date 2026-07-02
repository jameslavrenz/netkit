#include "ops.hpp"
#include "active_kernel.hpp"
#include "reference_kernel.hpp"

namespace Ops
{
    bool IsElementwiseValid(const Tensor& A, const Tensor& B)
    {
        if (A.rank != B.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != B.shape[i])
                return false;
        }

        return true;
    }

    void Mul(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!IsElementwiseValid(A, B))
        {
            std::cout << "Mul error: shape mismatch\n";
            return;
        }

        Kernels::Mul(A, B, C);
    }

    void MulScalar(const Tensor& A, float scalar, Tensor& C)
    {
        Kernels::MulScalar(A, scalar, C);
    }

    bool CheckSameShape2D(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != 2 || B.rank != 2 || C.rank != 2)
            return false;

        if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1])
            return false;

        if (A.shape[0] != C.shape[0] || A.shape[1] != C.shape[1])
            return false;

        return true;
    }

    void MatAdd(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!CheckSameShape2D(A, B, C))
        {
            std::cout << "Shape mismatch in Add()\n";
            return;
        }

        Kernels::MatAdd(A, B, C);
    }

    bool CheckSameShapeND(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != B.rank || A.rank != C.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != B.shape[i] || A.shape[i] != C.shape[i])
                return false;
        }

        return true;
    }

    void MatAddND(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!CheckSameShapeND(A, B, C))
        {
            std::cout << "Shape mismatch in AddND()\n";
            return;
        }

        Kernels::MatAddND(A, B, C);
    }

    bool IsMatMulValid(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        return (A.rank == 2 && B.rank == 2 && C.rank == 2 && A.shape[1] == B.shape[0] &&
                C.shape[0] == A.shape[0] && C.shape[1] == B.shape[1]);
    }

    void MatMul(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!IsMatMulValid(A, B, C))
        {
            std::cout << "MatMul shape mismatch\n";
            return;
        }

        Kernels::MatMul(A, B, C);
    }

    bool IsFullyConnectedValid(const Tensor& input, const Tensor& kernel, const Tensor& output)
    {
        return input.rank == 2 && kernel.rank == 2 && output.rank == 2 && kernel.shape[1] == input.shape[1] &&
               output.shape[0] == input.shape[0] && output.shape[1] == kernel.shape[0];
    }

    void FullyConnected(const Tensor& input, const Tensor& kernel, Tensor& output)
    {
        if (!IsFullyConnectedValid(input, kernel, output))
        {
            std::cout << "FullyConnected shape mismatch\n";
            return;
        }

        ReferenceKernel::FullyConnected(input, kernel, output);
    }

    bool IsElementwiseValidND(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != B.rank || A.rank != C.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != B.shape[i] || A.shape[i] != C.shape[i])
                return false;
        }

        return true;
    }

    void MulND(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!IsElementwiseValidND(A, B, C))
        {
            std::cout << "MulND shape mismatch\n";
            return;
        }

        Kernels::MulND(A, B, C);
    }

    bool IsUnaryOpValid(const Tensor& A, const Tensor& C)
    {
        if (A.rank != C.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != C.shape[i])
                return false;
        }

        return true;
    }

    void ReLU(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "ReLU shape mismatch\n";
            return;
        }

        Kernels::ReLU(A, C);
    }

    void Sigmoid(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "Sigmoid shape mismatch\n";
            return;
        }

        Kernels::Sigmoid(A, C);
    }

    void Tanh(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "Tanh shape mismatch\n";
            return;
        }

        Kernels::Tanh(A, C);
    }

    void LeakyReLU(const Tensor& A, Tensor& C, float alpha)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "LeakyReLU shape mismatch\n";
            return;
        }

        Kernels::LeakyReLU(A, C, alpha);
    }

    void ReLU6(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "ReLU6 shape mismatch\n";
            return;
        }

        Kernels::ReLU6(A, C);
    }

    void Softmax(const Tensor& A, Tensor& C)
    {
        Kernels::Softmax(A, C);
    }
}
