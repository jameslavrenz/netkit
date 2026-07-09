#include "mobilenetv4_uib.hpp"

#include "active_kernel.hpp"
#include "fused_kernel_ops.hpp"
#include "nk_op_detail.hpp"
#include "quant_ops.hpp"
#include "tensor_access.hpp"

#include <algorithm>
#include <cstddef>
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

namespace
{
    // Fold BatchNorm (y = x*scale + beta, per output channel) into the conv that
    // produced x. Conv output channel o is sum(W[o]*in) + conv_bias[o], so
    //   BN(o) = scale[o]*(sum(W[o]*in) + conv_bias[o]) + beta[o]
    //         = sum((scale[o]*W[o])*in) + (scale[o]*conv_bias[o] + beta[o]).
    // Weights are laid out as out_ch contiguous blocks of weights_per_out
    // (pointwise: in_c per out channel; depthwise: kh*kw per channel). Returns
    // the effective bias pointer to hand the kernel (the folded conv_bias, or
    // beta itself when the conv had no bias).
    float* FoldBnIntoConv(float* weights,
                          float* conv_bias,
                          const float* bn_scale,
                          const float* bn_bias,
                          uint32_t out_ch,
                          uint32_t weights_per_out)
    {
        for (uint32_t o = 0; o < out_ch; ++o)
        {
            const float s = bn_scale[o];
            float* w = weights + static_cast<std::size_t>(o) * weights_per_out;
            for (uint32_t k = 0; k < weights_per_out; ++k)
                w[k] *= s;
        }
        if (conv_bias)
        {
            for (uint32_t o = 0; o < out_ch; ++o)
                conv_bias[o] = bn_scale[o] * conv_bias[o] + bn_bias[o];
            return conv_bias;
        }
        // No conv bias: BN's beta becomes the (unscaled) fused bias.
        return const_cast<float*>(bn_bias);
    }
}

void MobileNetV4Uib::FoldBatchNorm()
{
    if (bn_folded)
        return;

    const uint32_t in_c = static_cast<uint32_t>(in_channels);
    const uint32_t out_c = static_cast<uint32_t>(out_channels);
    const uint32_t expand_c = expanded_channels();

    if (start_dw_kernel > 0 && start_bn_scale && start_bn_bias)
    {
        const uint32_t k2 = static_cast<uint32_t>(start_dw_kernel * start_dw_kernel);
        start_dw_bias =
            FoldBnIntoConv(start_dw_weights, start_dw_bias, start_bn_scale, start_bn_bias, in_c, k2);
    }

    if (expand_bn_scale && expand_bn_bias)
    {
        expand_bias =
            FoldBnIntoConv(expand_weights, expand_bias, expand_bn_scale, expand_bn_bias, expand_c, in_c);
    }

    if (middle_dw_kernel > 0 && middle_bn_scale && middle_bn_bias)
    {
        const uint32_t k2 = static_cast<uint32_t>(middle_dw_kernel * middle_dw_kernel);
        middle_dw_bias = FoldBnIntoConv(
            middle_dw_weights, middle_dw_bias, middle_bn_scale, middle_bn_bias, expand_c, k2);
    }

    if (proj_bn_scale && proj_bn_bias)
    {
        proj_bias =
            FoldBnIntoConv(proj_weights, proj_bias, proj_bn_scale, proj_bn_bias, out_c, expand_c);
    }

    bn_folded = true;
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

    // Fold BN into conv weights/bias once so the passes below need no standalone
    // BN and can fuse ReLU into the conv epilogue.
    FoldBatchNorm();

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
                               NetkitKernelActivation::ReLU,
                               next);

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
                                        NetkitKernelActivation::ReLU,
                                        next);

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
    }

    // Residual epilogue: MatAddND routes through CMSIS-DSP / CMSIS-NN / reference.
    if (has_residual())
    {
        Tensor residual = fused_ops::NhwcView(residual_buf, in_h, in_w, in_c);
        fused_ops::MatAddInPlace(output, residual);
    }
}

void MobileNetV4Uib::forward_quant(const int8_t* input, int8_t* output, uint32_t in_h, uint32_t in_w) const
{
    if (!quant_enabled || !input || !output || !scratch_i8)
        return;

    const uint32_t in_c = static_cast<uint32_t>(in_channels);
    const uint32_t expand_c = expanded_channels();

    uint32_t out_h = 0;
    uint32_t out_w = 0;
    output_spatial(in_h, in_w, out_h, out_w);

    const uint32_t max_spatial = in_h * in_w;
    int8_t* work_a = scratch_i8;
    int8_t* work_b = scratch_i8 + static_cast<std::size_t>(max_spatial) * expand_c;
    int8_t* residual_buf = work_b + static_cast<std::size_t>(max_spatial) * expand_c;

    if (has_residual())
        std::memcpy(residual_buf, input, static_cast<std::size_t>(in_h) * in_w * in_c);

    uint32_t cur_h = in_h;
    uint32_t cur_w = in_w;
    uint32_t cur_c = in_c;
    const int8_t* cur_data = input;
    int8_t* next_data = work_a;
    bool cur_in_work = false;

    if (start_dw_kernel > 0 && start_dw_weights_q && start_dw_bias_q)
    {
        const int pad = (start_dw_kernel - 1) / 2;
        const uint32_t next_h =
            CalcOutputDim(cur_h, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);
        const uint32_t next_w =
            CalcOutputDim(cur_w, start_dw_kernel, static_cast<int>(start_dw_stride()), pad);

        QuantOps::DepthwiseConv2dNhwcQuant(cur_data,
                                           cur_h,
                                           cur_w,
                                           cur_c,
                                           start_dw_weights_q,
                                           start_dw_bias_q,
                                           start_dw_kernel,
                                           start_dw_kernel,
                                           static_cast<int>(start_dw_stride()),
                                           pad,
                                           pad,
                                           pad,
                                           pad,
                                           start_dw_quant,
                                           false,
                                           next_data);

        cur_h = next_h;
        cur_w = next_w;
        cur_data = next_data;
        cur_in_work = true;
    }

    {
        int8_t* expand_out = cur_in_work ? work_b : work_a;
        QuantOps::Conv2dNhwcQuant(cur_data,
                                  cur_h,
                                  cur_w,
                                  cur_c,
                                  expand_weights_q,
                                  expand_bias_q,
                                  1,
                                  1,
                                  0,
                                  0,
                                  0,
                                  0,
                                  static_cast<int>(expand_c),
                                  expand_quant,
                                  true,
                                  expand_out);

        cur_c = expand_c;
        cur_data = expand_out;
        cur_in_work = expand_out == work_a;
    }

    if (middle_dw_kernel > 0 && middle_dw_weights_q && middle_dw_bias_q)
    {
        const int pad = (middle_dw_kernel - 1) / 2;
        const uint32_t next_h =
            CalcOutputDim(cur_h, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);
        const uint32_t next_w =
            CalcOutputDim(cur_w, middle_dw_kernel, static_cast<int>(middle_dw_stride()), pad);

        int8_t* middle_out = cur_in_work ? work_b : work_a;
        QuantOps::DepthwiseConv2dNhwcQuant(cur_data,
                                           cur_h,
                                           cur_w,
                                           cur_c,
                                           middle_dw_weights_q,
                                           middle_dw_bias_q,
                                           middle_dw_kernel,
                                           middle_dw_kernel,
                                           static_cast<int>(middle_dw_stride()),
                                           pad,
                                           pad,
                                           pad,
                                           pad,
                                           middle_dw_quant,
                                           true,
                                           middle_out);

        cur_h = next_h;
        cur_w = next_w;
        cur_data = middle_out;
    }

    QuantOps::ResidualAddS8 residual{};
    const QuantOps::ResidualAddS8* residual_ptr = nullptr;
    if (has_residual())
    {
        residual.data = residual_buf;
        residual.scale = block_input_scale;
        residual.zero_point = block_input_zero_point;
        residual_ptr = &residual;
    }
    QuantOps::Conv2dNhwcQuant(cur_data,
                              cur_h,
                              cur_w,
                              cur_c,
                              proj_weights_q,
                              proj_bias_q,
                              1,
                              1,
                              0,
                              0,
                              0,
                              0,
                              out_channels,
                              proj_quant,
                              false,
                              output,
                              residual_ptr);
}
