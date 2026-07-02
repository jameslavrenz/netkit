#include "ops.hpp"
#include "netkit_backend.h"
#include <cfloat>

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

        if (netkit_cmsis_dsp_mult_f32(&A, &B, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        const float* b = static_cast<const float*>(B.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = a[i] * b[i];
        }
    }

    void MulScalar(const Tensor& A, float scalar, Tensor& C)
    {
        if (netkit_cmsis_dsp_scale_f32(&A, scalar, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = a[i] * scalar;
        }
    }

    bool CheckSameShape2D(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != 2 || B.rank != 2 || C.rank != 2)
        {
            return false;
        }

        if (A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1])
        {
            return false;
        }

        if (A.shape[0] != C.shape[0] || A.shape[1] != C.shape[1])
        {
            return false;
        }

        return true;
    }

    void MatAdd(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!CheckSameShape2D(A, B, C))
        {
            std::cout << "Shape mismatch in Add()\n";
            return;
        }

        if (netkit_cmsis_nn_add_f32(&A, &B, &C))
            return;

        if (netkit_cmsis_dsp_add_f32(&A, &B, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        const float* b = static_cast<const float*>(B.data);
        float* c = static_cast<float*>(C.data);

        uint32_t M = A.shape[0];
        uint32_t N = A.shape[1];

        for (uint32_t i = 0; i < M; i++)
        {
            for (uint32_t j = 0; j < N; j++)
            {
                uint32_t a_idx = i * A.stride[0] + j * A.stride[1];
                uint32_t b_idx = i * B.stride[0] + j * B.stride[1];
                uint32_t c_idx = i * C.stride[0] + j * C.stride[1];

                c[c_idx] = a[a_idx] + b[b_idx];
            }
        }
    }

    bool CheckSameShapeND(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != B.rank || A.rank != C.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != B.shape[i] || A.shape[i] != C.shape[i])
            {
                return false;
            }
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

        if (netkit_cmsis_nn_add_f32(&A, &B, &C))
            return;

        if (netkit_cmsis_dsp_add_f32(&A, &B, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        const float* b = static_cast<const float*>(B.data);
        float* c = static_cast<float*>(C.data);

        uint32_t total = A.num_elements;

        for (uint32_t idx = 0; idx < total; idx++)
        {
            c[idx] = a[idx] + b[idx];
        }
    }

    bool IsMatMulValid(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        return (A.rank == 2 && B.rank == 2 && C.rank == 2 &&
                A.shape[1] == B.shape[0] &&
                C.shape[0] == A.shape[0] &&
                C.shape[1] == B.shape[1]);
    }

    void MatMul(const Tensor& A, const Tensor& B, Tensor& C)
    {
        if (!IsMatMulValid(A, B, C))
        {
            std::cout << "MatMul shape mismatch\n";
            return;
        }

        if (netkit_cmsis_dsp_mat_mul_f32(&A, &B, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        const float* b = static_cast<const float*>(B.data);
        float* c = static_cast<float*>(C.data);

        uint32_t M = A.shape[0];
        uint32_t K = A.shape[1];
        uint32_t N = B.shape[1];

        for (uint32_t i = 0; i < M; i++)
        {
            for (uint32_t j = 0; j < N; j++)
            {
                float sum = 0.0f;

                for (uint32_t k = 0; k < K; k++)
                {
                    uint32_t a_index = i * A.stride[0] + k * A.stride[1];
                    uint32_t b_index = k * B.stride[0] + j * B.stride[1];

                    sum += a[a_index] * b[b_index];
                }

                uint32_t c_index = i * C.stride[0] + j * C.stride[1];
                c[c_index] = sum;
            }
        }
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

        const float* in = static_cast<const float*>(input.data);
        const float* wt = static_cast<const float*>(kernel.data);
        float* out = static_cast<float*>(output.data);

        const uint32_t batch = input.shape[0];
        const uint32_t in_features = input.shape[1];
        const uint32_t out_features = kernel.shape[0];

        for (uint32_t b = 0; b < batch; ++b)
        {
            for (uint32_t oc = 0; oc < out_features; ++oc)
            {
                float sum = 0.0f;
                for (uint32_t ic = 0; ic < in_features; ++ic)
                {
                    const uint32_t in_index = b * input.stride[0] + ic * input.stride[1];
                    const uint32_t wt_index = oc * kernel.stride[0] + ic * kernel.stride[1];
                    sum += in[in_index] * wt[wt_index];
                }

                const uint32_t out_index = b * output.stride[0] + oc * output.stride[1];
                out[out_index] = sum;
            }
        }
    }

    bool IsElementwiseValidND(const Tensor& A, const Tensor& B, const Tensor& C)
    {
        if (A.rank != B.rank || A.rank != C.rank)
            return false;

        for (uint32_t i = 0; i < A.rank; i++)
        {
            if (A.shape[i] != B.shape[i] || A.shape[i] != C.shape[i])
            {
                return false;
            }
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

        if (netkit_cmsis_dsp_mult_f32(&A, &B, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        const float* b = static_cast<const float*>(B.data);
        float* c = static_cast<float*>(C.data);

        uint32_t total = A.num_elements;

        if (A.stride[0] == A.shape[1] && A.stride[1] == 1)
        {
            for (uint32_t i = 0; i < total; i++)
            {
                c[i] = a[i] * b[i];
            }
            return;
        }

        for (uint32_t idx = 0; idx < total; idx++)
        {
            uint32_t a_idx = idx;
            uint32_t b_idx = idx;
            uint32_t c_idx = idx;

            c[c_idx] = a[a_idx] * b[b_idx];
        }
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

        if (netkit_cmsis_activation_forward(&A, &C, NETKIT_BACKEND_ACT_RELU, 0.0f))
            return;

        if (netkit_cmsis_dsp_clip_f32(&A, &C, 0.0f, FLT_MAX))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
        }
    }

    void Sigmoid(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "Sigmoid shape mismatch\n";
            return;
        }

        if (netkit_cmsis_activation_forward(&A, &C, NETKIT_BACKEND_ACT_SIGMOID, 0.0f))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = 1.0f / (1.0f + std::expf(-a[i]));
        }
    }

    void Tanh(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "Tanh shape mismatch\n";
            return;
        }

        if (netkit_cmsis_activation_forward(&A, &C, NETKIT_BACKEND_ACT_TANH, 0.0f))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = std::tanhf(a[i]);
        }
    }

    void LeakyReLU(const Tensor& A, Tensor& C, float alpha)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "LeakyReLU shape mismatch\n";
            return;
        }

        if (netkit_cmsis_activation_forward(&A, &C, NETKIT_BACKEND_ACT_LEAKY_RELU, alpha))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            c[i] = (a[i] > 0.0f) ? a[i] : alpha * a[i];
        }
    }

    void ReLU6(const Tensor& A, Tensor& C)
    {
        if (!IsUnaryOpValid(A, C))
        {
            std::cout << "ReLU6 shape mismatch\n";
            return;
        }

        if (netkit_cmsis_activation_forward(&A, &C, NETKIT_BACKEND_ACT_RELU6, 0.0f))
            return;

        if (netkit_cmsis_dsp_clip_f32(&A, &C, 0.0f, 6.0f))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        for (uint32_t i = 0; i < A.num_elements; i++)
        {
            float x = a[i];

            if (x < 0.0f)
            {
                c[i] = 0.0f;
            }
            else if (x > 6.0f)
            {
                c[i] = 6.0f;
            }
            else
            {
                c[i] = x;
            }
        }
    }

    void Softmax(const Tensor& A, Tensor& C)
    {
        if (netkit_cmsis_softmax_forward(&A, &C))
            return;

        const float* a = static_cast<const float*>(A.data);
        float* c = static_cast<float*>(C.data);

        const uint32_t n = A.num_elements;

        float max_val = a[0];
        for (uint32_t i = 1; i < n; i++)
        {
            if (a[i] > max_val)
                max_val = a[i];
        }

        float sum = 0.0f;
        for (uint32_t i = 0; i < n; i++)
        {
            float e = std::expf(a[i] - max_val);
            c[i] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (uint32_t i = 0; i < n; i++)
        {
            c[i] *= inv_sum;
        }
    }
}
