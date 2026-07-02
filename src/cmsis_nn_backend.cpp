/*
 * CMSIS-NN adapter for netkit conv2d, max-pool, FC, activations (float32, NHWC).
 *
 * CMSIS-NN is Apache-2.0 — see third_party/CMSIS-NN/LICENSE.
 */
#include "cmsis_nn_kernel.hpp"
#include "netkit_config.h"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED

#include <arm_nnfunctions.h>

#include <alloca.h>
#include <cfloat>
#include <cstdint>

namespace
{
    bool cmsis_status_ok(arm_cmsis_nn_status status)
    {
        return status == ARM_CMSIS_NN_SUCCESS;
    }

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

    cmsis_nn_activation_f32 fused_activation_clamp(NetkitKernelActivation activation)
    {
        cmsis_nn_activation_f32 clamp = {.min = -FLT_MAX, .max = FLT_MAX};
        if (activation == NetkitKernelActivation::ReLU)
            clamp.min = 0.0f;
        else if (activation == NetkitKernelActivation::ReLU6)
        {
            clamp.min = 0.0f;
            clamp.max = 6.0f;
        }
        return clamp;
    }

    arm_nn_activation_type_flt map_activation_type(NetkitKernelActivation activation)
    {
        switch (activation)
        {
            case NetkitKernelActivation::ReLU:
                return ARM_NN_FLT_ACT_RELU;
            case NetkitKernelActivation::Sigmoid:
                return ARM_NN_FLT_ACT_SIGMOID;
            case NetkitKernelActivation::Tanh:
                return ARM_NN_FLT_ACT_TANH;
            case NetkitKernelActivation::LeakyReLU:
                return ARM_NN_FLT_ACT_LEAKY_RELU;
            case NetkitKernelActivation::ReLU6:
                return ARM_NN_FLT_ACT_RELU6;
            default:
                return ARM_NN_FLT_ACT_NONE;
        }
    }
}

bool CmsisNnKernel::TryConv2dForward(const Tensor& input,
                                     float* weights,
                                     float* bias,
                                     int kernel_size,
                                     int stride,
                                     int pad_h,
                                     int pad_w,
                                     int in_channels,
                                     int out_channels,
                                     NetkitKernelActivation fuse_activation,
                                     Tensor& output)
{
    if (!weights || input.rank != 3 || output.rank != 3)
        return false;

    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    if (static_cast<int>(input.shape[2]) != in_channels ||
        static_cast<int>(output.shape[2]) != out_channels)
        return false;

    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = pad_w, .h = pad_h},
        .stride = {.w = stride, .h = stride},
        .dilation = {.w = 1, .h = 1},
        .activation = fused_activation_clamp(fuse_activation),
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(in_h),
        .w = static_cast<int32_t>(in_w),
        .c = in_channels,
    };
    const cmsis_nn_dims filter_dims = {
        .n = out_channels,
        .h = kernel_size,
        .w = kernel_size,
        .c = in_channels,
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .w = 1, .h = 1, .c = out_channels};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(out_h),
        .w = static_cast<int32_t>(out_w),
        .c = out_channels,
    };

    const int32_t buf_size =
        arm_convolve_wrapper_f32_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    cmsis_nn_context ctx = {0};
    if (buf_size > 0)
    {
        if (buf_size > 262144)
            return false;
        ctx.buf = alloca(static_cast<size_t>(buf_size));
        ctx.size = buf_size;
    }

    return cmsis_status_ok(arm_convolve_wrapper_f32(&ctx,
                                                    &conv_params,
                                                    &input_dims,
                                                    in,
                                                    &filter_dims,
                                                    weights,
                                                    &bias_dims,
                                                    bias,
                                                    &output_dims,
                                                    out));
}

bool CmsisNnKernel::TryMaxPool2dForward(const Tensor& input,
                                        int pool_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        NetkitKernelActivation fuse_activation,
                                        Tensor& output)
{
    if (input.rank != 3 || output.rank != 3)
        return false;

    const cmsis_nn_pool_params_f32 pool_params = {
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .activation = fused_activation_clamp(fuse_activation),
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(input.shape[0]),
        .w = static_cast<int32_t>(input.shape[1]),
        .c = static_cast<int32_t>(input.shape[2]),
    };
    const cmsis_nn_dims filter_dims = {.n = 1, .h = pool_size, .w = pool_size, .c = 1};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(output.shape[0]),
        .w = static_cast<int32_t>(output.shape[1]),
        .c = static_cast<int32_t>(output.shape[2]),
    };

    cmsis_nn_context ctx = {0};
    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(arm_max_pool_f32(&ctx,
                                            &pool_params,
                                            &input_dims,
                                            in,
                                            &filter_dims,
                                            &output_dims,
                                            out));
}

bool CmsisNnKernel::TryAvgPool2dForward(const Tensor& input,
                                        int pool_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        Tensor& output)
{
    if (input.rank != 3 || output.rank != 3)
        return false;

    const cmsis_nn_pool_params_f32 pool_params = {
        .stride = {.w = stride, .h = stride},
        .padding = {.w = pad_w, .h = pad_h},
        .activation = {.min = -FLT_MAX, .max = FLT_MAX},
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(input.shape[0]),
        .w = static_cast<int32_t>(input.shape[1]),
        .c = static_cast<int32_t>(input.shape[2]),
    };
    const cmsis_nn_dims filter_dims = {.n = 1, .h = pool_size, .w = pool_size, .c = 1};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(output.shape[0]),
        .w = static_cast<int32_t>(output.shape[1]),
        .c = static_cast<int32_t>(output.shape[2]),
    };

    cmsis_nn_context ctx = {0};
    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(arm_avg_pool_f32(&ctx,
                                            &pool_params,
                                            &input_dims,
                                            in,
                                            &filter_dims,
                                            &output_dims,
                                            out));
}

bool CmsisNnKernel::TryBatchNorm2dForward(const Tensor& input,
                                          const float* scale,
                                          const float* bias,
                                          int channels,
                                          Tensor& output)
{
    if (!scale || !bias || input.rank != 3 || output.rank != 3)
        return false;

    if (static_cast<int>(input.shape[2]) != channels || static_cast<int>(output.shape[2]) != channels)
        return false;

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(input.shape[0]),
        .w = static_cast<int32_t>(input.shape[1]),
        .c = channels,
    };

    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(
        arm_batch_norm_f32(in, out, scale, bias, &input_dims, ARM_NN_LAYOUT_NHWC));
}

bool CmsisNnKernel::TryFullyConnectedWithBias(const Tensor& input,
                                              const Tensor& kernel,
                                              const Tensor& bias,
                                              NetkitKernelActivation fuse_activation,
                                              Tensor& output)
{
    if (input.rank != 2 || kernel.rank != 2 || bias.rank != 2 || output.rank != 2)
        return false;

    const uint32_t batch = input.shape[0];
    const uint32_t in_features = input.shape[1];
    const uint32_t out_features = kernel.shape[0];

    if (kernel.shape[1] != in_features || output.shape[0] != batch || output.shape[1] != out_features ||
        bias.shape[0] != 1 || bias.shape[1] != out_features)
        return false;

    const cmsis_nn_fc_params_f32 fc_params = {
        .activation = fused_activation_clamp(fuse_activation),
        .weight_format = ARM_NN_WEIGHT_FORMAT_STANDARD,
    };

    const cmsis_nn_dims input_dims = {
        .n = static_cast<int32_t>(batch),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(in_features),
    };
    const cmsis_nn_dims filter_dims = {
        .n = static_cast<int32_t>(in_features),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(out_features),
    };
    const cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = static_cast<int32_t>(out_features)};
    const cmsis_nn_dims output_dims = {
        .n = static_cast<int32_t>(batch),
        .h = 1,
        .w = 1,
        .c = static_cast<int32_t>(out_features),
    };

    cmsis_nn_context ctx = {0};
    const float* in = static_cast<const float*>(input.data);
    const float* wt = static_cast<const float*>(kernel.data);
    const float* b = static_cast<const float*>(bias.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(arm_fully_connected_f32(&ctx,
                                                   &fc_params,
                                                   &input_dims,
                                                   in,
                                                   &filter_dims,
                                                   wt,
                                                   &bias_dims,
                                                   b,
                                                   &output_dims,
                                                   out,
                                                   ARM_NN_LAYOUT_NHWC));
}

bool CmsisNnKernel::TryActivationForward(const Tensor& input,
                                         Tensor& output,
                                         NetkitKernelActivation activation,
                                         float leaky_alpha)
{
    if (input.num_elements != output.num_elements)
        return false;
    if (activation == NetkitKernelActivation::None || activation == NetkitKernelActivation::Softmax)
        return false;

    const arm_nn_activation_type_flt type = map_activation_type(activation);
    if (type == ARM_NN_FLT_ACT_NONE)
        return false;

    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(arm_nn_activation_f32(in,
                                                 out,
                                                 static_cast<int32_t>(input.num_elements),
                                                 type,
                                                 leaky_alpha));
}

bool CmsisNnKernel::TrySoftmaxForward(const Tensor& input, Tensor& output)
{
    if (input.num_elements != output.num_elements)
        return false;

    const float* in = static_cast<const float*>(input.data);
    float* out = static_cast<float*>(output.data);

    return cmsis_status_ok(arm_softmax_f32(in,
                                           1,
                                           static_cast<int32_t>(input.num_elements),
                                           out));
}

bool CmsisNnKernel::TryMatAdd(const Tensor& a, const Tensor& b, Tensor& c)
{
    if (a.num_elements != b.num_elements || a.num_elements != c.num_elements)
        return false;
    if (!tensor_is_dense(a) || !tensor_is_dense(b) || !tensor_is_dense(c))
        return false;

    return cmsis_status_ok(arm_elementwise_add_f32(static_cast<const float*>(a.data),
                                                   static_cast<const float*>(b.data),
                                                   static_cast<float*>(c.data),
                                                   -FLT_MAX,
                                                   FLT_MAX,
                                                   static_cast<int32_t>(a.num_elements)));
}

#endif /* NETKIT_USE_CMSIS_NN */
