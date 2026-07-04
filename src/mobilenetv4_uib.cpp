#include "mobilenetv4_uib.hpp"

#include "active_kernel.hpp"
#include "fused_kernel_ops.hpp"
#include "nk_op_detail.hpp"
#include "tensor_access.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{
    using nk_op_detail::CalcOutputDim;
}

uint32_t MobileNetV4Uib::MakeDivisible(float value, uint32_t divisor)
{
    const uint32_t rounded =
        static_cast<uint32_t>(value + static_cast<float>(divisor) * 0.5f) / divisor * divisor;
    const uint32_t min_value = divisor;
    uint32_t result = std::max(min_value, rounded);
    if (result < static_cast<uint32_t>(0.9f * value))
        result += divisor;
    return result;
}

uint32_t MobileNetV4Uib::expanded_channels() const
{
    return MakeDivisible(static_cast<float>(in_channels) * expand_ratio, 8);
}

uint32_t MobileNetV4Uib::start_dw_stride() const
{
    if (start_dw_kernel <= 0)
        return 1;
    if (middle_dw_downsample && middle_dw_kernel > 0)
        return 1;
    return static_cast<uint32_t>(stride);
}

uint32_t MobileNetV4Uib::middle_dw_stride() const
{
    if (middle_dw_kernel <= 0)
        return 1;
    return middle_dw_downsample ? static_cast<uint32_t>(stride) : 1u;
}

void MobileNetV4Uib::output_spatial(uint32_t in_h, uint32_t in_w, uint32_t& out_h, uint32_t& out_w) const
{
    out_h = in_h;
    out_w = in_w;

    if (start_dw_kernel > 0)
    {
        const int pad = (start_dw_kernel - 1) / 2;
        out_h = CalcOutputDim(out_h, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);
        out_w = CalcOutputDim(out_w, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);
    }

    if (middle_dw_kernel > 0)
    {
        const int pad = (middle_dw_kernel - 1) / 2;
        out_h = CalcOutputDim(out_h, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);
        out_w = CalcOutputDim(out_w, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);
    }
}

bool MobileNetV4Uib::has_residual() const
{
    return stride == 1 && in_channels == out_channels;
}

void MobileNetV4Uib::forward(const Tensor& input, Tensor& output)
{
    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    const uint32_t in_c = static_cast<uint32_t>(in_channels);
    const uint32_t expand_c = expanded_channels();

    uint32_t out_h = 0;
    uint32_t out_w = 0;
    output_spatial(in_h, in_w, out_h, out_w);

    const uint32_t max_spatial = in_h * in_w;
    const uint32_t required_scratch =
        2u * max_spatial * expand_c + (has_residual() ? max_spatial * in_c : 0u);

    if (!scratch || scratch_elems < required_scratch || !output.data)
        return;

    float* work_a = scratch;
    float* work_b = scratch + static_cast<std::size_t>(max_spatial) * expand_c;
    float* residual_buf = work_b + static_cast<std::size_t>(max_spatial) * expand_c;

    if (has_residual())
        std::memcpy(residual_buf, tensor_data_f32(const_cast<Tensor&>(input)),
                    static_cast<std::size_t>(input.num_elements) * sizeof(float));

    uint32_t cur_h = in_h;
    uint32_t cur_w = in_w;
    uint32_t cur_c = in_c;
    float* cur_data = tensor_data_f32(const_cast<Tensor&>(input));
    float* next_data = work_a;
    bool cur_in_work = false;

    if (start_dw_kernel > 0)
    {
        const int pad = (start_dw_kernel - 1) / 2;
        const uint32_t next_h =
            CalcOutputDim(cur_h, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);
        const uint32_t next_w =
            CalcOutputDim(cur_w, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);

        Tensor cur = fused_ops::NhwcView(cur_data, cur_h, cur_w, cur_c);
        Tensor next = fused_ops::NhwcView(next_data, next_h, next_w, cur_c);
        Kernels::DepthwiseConv2dForward(cur,
                                        start_dw_weights,
                                        start_dw_bias,
                                        start_dw_kernel,
                                        start_dw_kernel,
                                        static_cast<int>(start_dw_stride()),
                                        pad,
                                        pad,
                                        pad,
                                        pad,
                                        in_channels,
                                        NetkitKernelActivation::None,
                                        next);
        fused_ops::BatchNormInPlace(next, in_channels, start_bn_scale, start_bn_bias);

        cur_h = next_h;
        cur_w = next_w;
        cur_data = next_data;
        cur_in_work = true;
    }

    {
        float* expand_out = cur_in_work ? work_b : work_a;
        Tensor cur = fused_ops::NhwcView(cur_data, cur_h, cur_w, cur_c);
        Tensor next = fused_ops::NhwcView(expand_out, cur_h, cur_w, expand_c);
        Kernels::Conv2dForward(cur,
                               expand_weights,
                               expand_bias,
                               1,
                               1,
                               0,
                               0,
                               static_cast<int>(cur_c),
                               static_cast<int>(expand_c),
                               NetkitKernelActivation::None,
                               next);
        fused_ops::BatchNormInPlace(next, static_cast<int>(expand_c), expand_bn_scale, expand_bn_bias);
        fused_ops::ReluInPlace(next);

        cur_c = expand_c;
        cur_data = expand_out;
        cur_in_work = expand_out == work_a;
    }

    if (middle_dw_kernel > 0)
    {
        const int pad = (middle_dw_kernel - 1) / 2;
        const uint32_t next_h =
            CalcOutputDim(cur_h, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);
        const uint32_t next_w =
            CalcOutputDim(cur_w, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);

        float* middle_out = cur_in_work ? work_b : work_a;
        Tensor cur = fused_ops::NhwcView(cur_data, cur_h, cur_w, cur_c);
        Tensor next = fused_ops::NhwcView(middle_out, next_h, next_w, cur_c);
        Kernels::DepthwiseConv2dForward(cur,
                                        middle_dw_weights,
                                        middle_dw_bias,
                                        middle_dw_kernel,
                                        middle_dw_kernel,
                                        static_cast<int>(middle_dw_stride()),
                                        pad,
                                        pad,
                                        pad,
                                        pad,
                                        static_cast<int>(expand_c),
                                        NetkitKernelActivation::None,
                                        next);
        fused_ops::BatchNormInPlace(next, static_cast<int>(expand_c), middle_bn_scale, middle_bn_bias);
        fused_ops::ReluInPlace(next);

        cur_h = next_h;
        cur_w = next_w;
        cur_data = middle_out;
    }

    {
        Tensor cur = fused_ops::NhwcView(cur_data, cur_h, cur_w, cur_c);
        Kernels::Conv2dForward(cur,
                               proj_weights,
                               proj_bias,
                               1,
                               1,
                               0,
                               0,
                               static_cast<int>(expand_c),
                               out_channels,
                               NetkitKernelActivation::None,
                               output);
        fused_ops::BatchNormInPlace(output, out_channels, proj_bn_scale, proj_bn_bias);
    }

    if (has_residual())
    {
        Tensor residual = fused_ops::NhwcView(residual_buf, in_h, in_w, in_c);
        fused_ops::MatAddInPlace(output, residual);
    }
}
