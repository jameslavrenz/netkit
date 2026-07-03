#include "ops.hpp"
#include "active_kernel.hpp"

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
            NETKIT_LOG("Mul error: shape mismatch\n");
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
            NETKIT_LOG("Shape mismatch in Add()\n");
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
            NETKIT_LOG("Shape mismatch in AddND()\n");
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
            NETKIT_LOG("MatMul shape mismatch\n");
            return;
        }

        Kernels::MatMul(A, B, C);
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
            NETKIT_LOG("MulND shape mismatch\n");
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
            NETKIT_LOG("ReLU shape mismatch\n");
            return;
        }

        Kernels::ReLU(A, C);
    }

    void Sigmoid(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            NETKIT_LOG("Sigmoid shape mismatch\n");
            return;
        }

        Kernels::Sigmoid(A, C);
    }

    void Tanh(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            NETKIT_LOG("Tanh shape mismatch\n");
            return;
        }

        Kernels::Tanh(A, C);
    }

    void LeakyReLU(const Tensor& A, Tensor& C, float alpha)
    {
        if (!IsUnaryOpValid(A, C))
        {
            NETKIT_LOG("LeakyReLU shape mismatch\n");
            return;
        }

        Kernels::LeakyReLU(A, C, alpha);
    }

    void ReLU6(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            NETKIT_LOG("ReLU6 shape mismatch\n");
            return;
        }

        Kernels::ReLU6(A, C);
    }

    void Softmax(const Tensor& A, Tensor& C)
    {
        Kernels::Softmax(A, C);
    }
}
