/*
 * CMSIS-DSP adapter for netkit vector/matrix ops (float32).
 *
 * CMSIS-DSP is Apache-2.0 — see third_party/CMSIS-DSP/LICENSE.
 */
#include "cmsis_dsp_kernel.hpp"

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP

#include <arm_math.h>
#include <cstdint>

namespace
{
    bool tensor_is_dense(const Tensor& tensor)
    {
        if (tensor.rank == 0)
            return false;

        uint32_t expected_stride = 1;
        for (int i = static_cast<int>(tensor.rank) - 1; i >= 0; --i)
        {
            if (tensor.stride[static_cast<uint32_t>(i)] != expected_stride)
                return false;
            expected_stride *= tensor.shape[static_cast<uint32_t>(i)];
        }
        return true;
    }

    bool matrices_are_dense_row_major(const Tensor& a, const Tensor& b, const Tensor& c)
    {
        return a.rank == 2 && b.rank == 2 && c.rank == 2 && tensor_is_dense(a) && tensor_is_dense(b) &&
               tensor_is_dense(c);
    }
}

bool CmsisDspKernel::TryMatAdd(const Tensor& a, const Tensor& b, Tensor& c)
{
    if (a.num_elements != b.num_elements || a.num_elements != c.num_elements)
        return false;
    if (!tensor_is_dense(a) || !tensor_is_dense(b) || !tensor_is_dense(c))
        return false;

    arm_add_f32(static_cast<const float32_t*>(a.data),
                static_cast<const float32_t*>(b.data),
                static_cast<float32_t*>(c.data),
                static_cast<uint32_t>(a.num_elements));
    return true;
}

bool CmsisDspKernel::TryMul(const Tensor& a, const Tensor& b, Tensor& c)
{
    if (a.num_elements != b.num_elements || a.num_elements != c.num_elements)
        return false;
    if (!tensor_is_dense(a) || !tensor_is_dense(b) || !tensor_is_dense(c))
        return false;

    arm_mult_f32(static_cast<const float32_t*>(a.data),
                 static_cast<const float32_t*>(b.data),
                 static_cast<float32_t*>(c.data),
                 static_cast<uint32_t>(a.num_elements));
    return true;
}

bool CmsisDspKernel::TryMulScalar(const Tensor& a, float scalar, Tensor& c)
{
    if (a.num_elements != c.num_elements)
        return false;
    if (!tensor_is_dense(a) || !tensor_is_dense(c))
        return false;

    arm_scale_f32(static_cast<const float32_t*>(a.data),
                  scalar,
                  static_cast<float32_t*>(c.data),
                  static_cast<uint32_t>(a.num_elements));
    return true;
}

bool CmsisDspKernel::TryMatMul(const Tensor& a, const Tensor& b, Tensor& c)
{
    if (!matrices_are_dense_row_major(a, b, c))
        return false;
    if (a.shape[1] != b.shape[0] || c.shape[0] != a.shape[0] || c.shape[1] != b.shape[1])
        return false;

    arm_matrix_instance_f32 src_a{};
    arm_matrix_instance_f32 src_b{};
    arm_matrix_instance_f32 dst{};

    arm_mat_init_f32(&src_a,
                     static_cast<uint16_t>(a.shape[0]),
                     static_cast<uint16_t>(a.shape[1]),
                     const_cast<float32_t*>(static_cast<const float32_t*>(a.data)));
    arm_mat_init_f32(&src_b,
                     static_cast<uint16_t>(b.shape[0]),
                     static_cast<uint16_t>(b.shape[1]),
                     const_cast<float32_t*>(static_cast<const float32_t*>(b.data)));
    arm_mat_init_f32(&dst,
                     static_cast<uint16_t>(c.shape[0]),
                     static_cast<uint16_t>(c.shape[1]),
                     static_cast<float32_t*>(c.data));

    return arm_mat_mult_f32(&src_a, &src_b, &dst) == ARM_MATH_SUCCESS;
}

bool CmsisDspKernel::TryFullyConnectedWithBias(const Tensor& input,
                                               const Tensor& kernel,
                                               const Tensor& bias,
                                               Tensor& output)
{
    if (input.rank != 2 || kernel.rank != 2 || bias.rank != 2 || output.rank != 2)
        return false;
    if (!matrices_are_dense_row_major(input, kernel, output) || !tensor_is_dense(bias))
        return false;

    const uint32_t batch = input.shape[0];
    const uint32_t in_features = input.shape[1];
    const uint32_t out_features = kernel.shape[0];

    if (kernel.shape[1] != in_features || output.shape[0] != batch || output.shape[1] != out_features ||
        bias.shape[0] != 1 || bias.shape[1] != out_features)
        return false;

    arm_matrix_instance_f32 weight_mat{};
    arm_mat_init_f32(&weight_mat,
                     static_cast<uint16_t>(out_features),
                     static_cast<uint16_t>(in_features),
                     const_cast<float32_t*>(static_cast<const float32_t*>(kernel.data)));

    const float32_t* in = static_cast<const float32_t*>(input.data);
    float32_t* out = static_cast<float32_t*>(output.data);
    const float32_t* b = static_cast<const float32_t*>(bias.data);

    for (uint32_t row = 0; row < batch; ++row)
    {
        arm_mat_vec_mult_f32(&weight_mat, in + row * in_features, out + row * out_features);
        arm_add_f32(out + row * out_features, b, out + row * out_features, static_cast<uint32_t>(out_features));
    }

    return true;
}

bool CmsisDspKernel::TryClip(const Tensor& input, Tensor& output, float low, float high)
{
    if (input.num_elements != output.num_elements)
        return false;
    if (!tensor_is_dense(input) || !tensor_is_dense(output))
        return false;

    arm_clip_f32(static_cast<const float32_t*>(input.data),
                 static_cast<float32_t*>(output.data),
                 low,
                 high,
                 static_cast<uint32_t>(input.num_elements));
    return true;
}

bool CmsisDspKernel::TryBatchNorm2dForward(const Tensor& input,
                                           const float* scale,
                                           const float* bias,
                                           int channels,
                                           Tensor& output)
{
    if (!scale || !bias || input.rank != 3 || output.rank != 3)
        return false;
    if (!tensor_is_dense(input) || !tensor_is_dense(output))
        return false;
    if (static_cast<int>(input.shape[2]) != channels || static_cast<int>(output.shape[2]) != channels)
        return false;
    if (input.shape[0] != output.shape[0] || input.shape[1] != output.shape[1])
        return false;
    if (channels <= 0)
        return false;

    const uint32_t channel_count = static_cast<uint32_t>(channels);
    const uint32_t spatial = input.shape[0] * input.shape[1];
    const float32_t* scale_vec = static_cast<const float32_t*>(scale);
    const float32_t* bias_vec = static_cast<const float32_t*>(bias);

    const float32_t* in = static_cast<const float32_t*>(input.data);
    float32_t* out = static_cast<float32_t*>(output.data);

    for (uint32_t p = 0; p < spatial; ++p)
    {
        const float32_t* in_row = in + p * channel_count;
        float32_t* out_row = out + p * channel_count;
        arm_mult_f32(in_row, scale_vec, out_row, channel_count);
        arm_add_f32(out_row, bias_vec, out_row, channel_count);
    }

    return true;
}

#endif /* NETKIT_USE_CMSIS_DSP */
