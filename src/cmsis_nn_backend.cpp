/*
 * CMSIS-NN adapter for netkit conv2d, max-pool, FC, activations (float32, NHWC).
 *
 * CMSIS-NN is Apache-2.0 — see third_party/CMSIS-NN/LICENSE.
 */
#include "netkit_backend.h"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN

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

    bool tensor_is_dense(const Tensor* tensor)
    {
        if (!tensor || tensor->rank == 0)
            return false;

        uint32_t expected_stride = 1;
        for (int i = static_cast<int>(tensor->rank) - 1; i >= 0; --i)
        {
            if (tensor->stride[static_cast<uint32_t>(i)] != expected_stride)
                return false;
            expected_stride *= tensor->shape[static_cast<uint32_t>(i)];
        }
        return true;
    }

    cmsis_nn_activation_f32 fused_activation_clamp(NetkitBackendActivation activation)
    {
        cmsis_nn_activation_f32 clamp = {.min = -FLT_MAX, .max = FLT_MAX};
        if (activation == NETKIT_BACKEND_ACT_RELU)
            clamp.min = 0.0f;
        else if (activation == NETKIT_BACKEND_ACT_RELU6)
        {
            clamp.min = 0.0f;
            clamp.max = 6.0f;
        }
        return clamp;
    }

    arm_nn_activation_type_flt map_activation_type(NetkitBackendActivation activation)
    {
        switch (activation)
        {
            case NETKIT_BACKEND_ACT_RELU:
                return ARM_NN_FLT_ACT_RELU;
            case NETKIT_BACKEND_ACT_SIGMOID:
                return ARM_NN_FLT_ACT_SIGMOID;
            case NETKIT_BACKEND_ACT_TANH:
                return ARM_NN_FLT_ACT_TANH;
            case NETKIT_BACKEND_ACT_LEAKY_RELU:
                return ARM_NN_FLT_ACT_LEAKY_RELU;
            case NETKIT_BACKEND_ACT_RELU6:
                return ARM_NN_FLT_ACT_RELU6;
            default:
                return ARM_NN_FLT_ACT_NONE;
        }
    }
}

extern "C" int netkit_cmsis_conv2d_forward(const Tensor* input,
                                           const float* weights,
                                           const float* bias,
                                           int kernel_size,
                                           int stride,
                                           int in_channels,
                                           int out_channels,
                                           NetkitBackendActivation fuse_activation,
                                           Tensor* output)
{
    if (!input || !output || !weights || input->rank != 3 || output->rank != 3)
        return 0;

    const uint32_t in_h = input->shape[0];
    const uint32_t in_w = input->shape[1];
    const uint32_t out_h = output->shape[0];
    const uint32_t out_w = output->shape[1];

    if (static_cast<int>(input->shape[2]) != in_channels ||
        static_cast<int>(output->shape[2]) != out_channels)
        return 0;

    const float* in = static_cast<const float*>(input->data);
    float* out = static_cast<float*>(output->data);

    const cmsis_nn_conv_params_f32 conv_params = {
        .padding = {.w = 0, .h = 0},
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
            return 0;
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
                                                    out))
               ? 1
               : 0;
}

extern "C" int netkit_cmsis_max_pool2d_forward(const Tensor* input,
                                               int pool_size,
                                               int stride,
                                               NetkitBackendActivation fuse_activation,
                                               Tensor* output)
{
    if (!input || !output || input->rank != 3 || output->rank != 3)
        return 0;

    const cmsis_nn_pool_params_f32 pool_params = {
        .stride = {.w = stride, .h = stride},
        .padding = {.w = 0, .h = 0},
        .activation = fused_activation_clamp(fuse_activation),
    };

    const cmsis_nn_dims input_dims = {
        .n = 1,
        .h = static_cast<int32_t>(input->shape[0]),
        .w = static_cast<int32_t>(input->shape[1]),
        .c = static_cast<int32_t>(input->shape[2]),
    };
    const cmsis_nn_dims filter_dims = {.n = 1, .h = pool_size, .w = pool_size, .c = 1};
    const cmsis_nn_dims output_dims = {
        .n = 1,
        .h = static_cast<int32_t>(output->shape[0]),
        .w = static_cast<int32_t>(output->shape[1]),
        .c = static_cast<int32_t>(output->shape[2]),
    };

    cmsis_nn_context ctx = {0};
    const float* in = static_cast<const float*>(input->data);
    float* out = static_cast<float*>(output->data);

    return cmsis_status_ok(arm_max_pool_f32(&ctx,
                                            &pool_params,
                                            &input_dims,
                                            in,
                                            &filter_dims,
                                            &output_dims,
                                            out))
               ? 1
               : 0;
}

extern "C" int netkit_cmsis_fully_connected_forward(const Tensor* input,
                                                    const Tensor* kernel,
                                                    const Tensor* bias,
                                                    NetkitBackendActivation fuse_activation,
                                                    Tensor* output)
{
    if (!input || !kernel || !output || !bias || input->rank != 2 || kernel->rank != 2 || bias->rank != 2 ||
        output->rank != 2)
        return 0;

    const uint32_t batch = input->shape[0];
    const uint32_t in_features = input->shape[1];
    const uint32_t out_features = kernel->shape[0];

    if (kernel->shape[1] != in_features || output->shape[0] != batch || output->shape[1] != out_features ||
        bias->shape[0] != 1 || bias->shape[1] != out_features)
        return 0;

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
    const float* in = static_cast<const float*>(input->data);
    const float* wt = static_cast<const float*>(kernel->data);
    const float* b = static_cast<const float*>(bias->data);
    float* out = static_cast<float*>(output->data);

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
                                                   ARM_NN_LAYOUT_NHWC))
               ? 1
               : 0;
}

extern "C" int netkit_cmsis_activation_forward(const Tensor* input,
                                               Tensor* output,
                                               NetkitBackendActivation activation,
                                               float leaky_alpha)
{
    if (!input || !output || input->num_elements != output->num_elements)
        return 0;
    if (activation == NETKIT_BACKEND_ACT_NONE || activation == NETKIT_BACKEND_ACT_SOFTMAX)
        return 0;

    const arm_nn_activation_type_flt type = map_activation_type(activation);
    if (type == ARM_NN_FLT_ACT_NONE)
        return 0;

    const float* in = static_cast<const float*>(input->data);
    float* out = static_cast<float*>(output->data);

    return cmsis_status_ok(arm_nn_activation_f32(in,
                                                 out,
                                                 static_cast<int32_t>(input->num_elements),
                                                 type,
                                                 leaky_alpha))
               ? 1
               : 0;
}

extern "C" int netkit_cmsis_softmax_forward(const Tensor* input, Tensor* output)
{
    if (!input || !output || input->num_elements != output->num_elements)
        return 0;

    const float* in = static_cast<const float*>(input->data);
    float* out = static_cast<float*>(output->data);

    return cmsis_status_ok(arm_softmax_f32(in,
                                           1,
                                           static_cast<int32_t>(input->num_elements),
                                           out))
               ? 1
               : 0;
}

extern "C" int netkit_cmsis_nn_add_f32(const Tensor* a, const Tensor* b, Tensor* c)
{
    if (!a || !b || !c || a->num_elements != b->num_elements || a->num_elements != c->num_elements)
        return 0;
    if (!tensor_is_dense(a) || !tensor_is_dense(b) || !tensor_is_dense(c))
        return 0;

    return cmsis_status_ok(arm_elementwise_add_f32(static_cast<const float*>(a->data),
                                                   static_cast<const float*>(b->data),
                                                   static_cast<float*>(c->data),
                                                   -FLT_MAX,
                                                   FLT_MAX,
                                                   static_cast<int32_t>(a->num_elements)))
               ? 1
               : 0;
}

#endif /* NETKIT_USE_CMSIS_NN */
