#include "quant_ops.hpp"

#include "cmsis_nn_quant.hpp"
#include "cmsis_dsp_util.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_integer.hpp"
#include "quant_trace.hpp"
#include "xnnpack_quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace QuantOps
{
    void FullyConnectedQuant(const int8_t* input,
                             uint32_t batch,
                             uint32_t in_features,
                             const int8_t* weights,
                             const int32_t* bias,
                             uint32_t out_features,
                             const NkFormat::MlpLayerQuantDesc& quant,
                             QuantInteger::QuantClamp clamp,
                             int8_t* output_int8)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryFullyConnectedQuant(input,
                                                 batch,
                                                 in_features,
                                                 weights,
                                                 bias,
                                                 out_features,
                                                 quant,
                                                 clamp == QuantInteger::QuantClamp::ReLU ||
                                                     clamp == QuantInteger::QuantClamp::ReLU6,
                                                 output_int8))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryFullyConnectedQuant(input,
                                                 batch,
                                                 in_features,
                                                 weights,
                                                 bias,
                                                 out_features,
                                                 quant,
                                                 clamp == QuantInteger::QuantClamp::ReLU ||
                                                     clamp == QuantInteger::QuantClamp::ReLU6,
                                                 output_int8))
        {
            return;
        }
#endif
        QuantTrace::RecordFcReference();

        // Int8→int8 only: integer multiply-by-quantized-multiplier (no float requantize/dequant).
        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == out_features;

        int32_t multiplier = 0;
        int32_t shift = 0;
        if (!per_channel)
        {
            if (!output_int8 ||
                !QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                        quant.weight_scale,
                                                        quant.output_scale,
                                                        &multiplier,
                                                        &shift))
            {
                return;
            }
        }
        else if (!output_int8)
        {
            return;
        }

        for (size_t b = 0; b < batch; ++b)
        {
            const int8_t* in_row = input + b * in_features;
            for (size_t oc = 0; oc < out_features; ++oc)
            {
                int32_t acc = bias[oc];
                const int8_t* wt_row = weights + oc * in_features;
                for (size_t ic = 0; ic < in_features; ++ic)
                {
                    const int32_t in_val = static_cast<int32_t>(in_row[ic]) - quant.input_zero_point;
                    const int32_t wt_val = static_cast<int32_t>(wt_row[ic]) - quant.weight_zero_point;
                    acc += in_val * wt_val;
                }

                int32_t m = multiplier;
                int32_t s = shift;
                if (per_channel)
                {
                    if (!QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                                quant.weight_channel_scales[oc],
                                                                quant.output_scale,
                                                                &m,
                                                                &s))
                    {
                        return;
                    }
                }
                output_int8[b * out_features + oc] = QuantInteger::RequantizeAccToInt8(
                    acc, m, s, quant.output_zero_point, clamp, quant.output_scale);
            }
        }
    }

    void ForwardQuantizedDense(const Tensor& input,
                               const Tensor& weights,
                               const Tensor& bias,
                               const LayerQuant& quant,
                               bool apply_relu,
                               bool apply_softmax,
                               int8_t* quant_scratch,
                               Tensor& output)
    {
        if (input.type != DataType::Int8 || output.type != DataType::Int8 || !input.data || !output.data)
            return;

        const uint32_t batch = input.shape[0];
        const uint32_t in_features = input.shape[1];
        const uint32_t out_features = weights.shape[0];

        const int8_t* input_i8 = static_cast<const int8_t*>(input.data);
        const int8_t* weight_q = static_cast<const int8_t*>(weights.data);
        const int32_t* bias_q = static_cast<const int32_t*>(bias.data);
        int8_t* logits_i8 = quant_scratch;

        if (apply_softmax)
        {
            FullyConnectedQuant(input_i8,
                                batch,
                                in_features,
                                weight_q,
                                bias_q,
                                out_features,
                                quant.params,
                                false,
                                logits_i8);

            const float logit_scale =
                quant.params.output_scale > 0.0f ? quant.params.output_scale : 1.0f;
            SoftmaxS8(logits_i8, out_features, logit_scale, static_cast<int8_t*>(output.data));
            return;
        }

        FullyConnectedQuant(input_i8,
                            batch,
                            in_features,
                            weight_q,
                            bias_q,
                            out_features,
                            quant.params,
                            apply_relu,
                            static_cast<int8_t*>(output.data));
    }

    void Conv2dNhwcQuant(const int8_t* input,
                         uint32_t in_h,
                         uint32_t in_w,
                         uint32_t in_c,
                         const int8_t* weights,
                         const int32_t* bias,
                         int kernel_size,
                         int stride,
                         int pad_h,
                         int pad_w,
                         int pad_h_end,
                         int pad_w_end,
                         int out_channels,
                         const NkFormat::MlpLayerQuantDesc& quant,
                         QuantInteger::QuantClamp clamp,
                         int8_t* output,
                         const ResidualAddS8* residual)
    {
        bool wrote = false;
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryConv2dNhwcQuant(input,
                                             in_h,
                                             in_w,
                                             in_c,
                                             weights,
                                             bias,
                                             kernel_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end,
                                             out_channels,
                                             quant,
                                             clamp == QuantInteger::QuantClamp::ReLU ||
                                                 clamp == QuantInteger::QuantClamp::ReLU6,
                                             output))
        {
            wrote = true;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (!wrote && CmsisNnQuant::TryConv2dNhwcQuant(input,
                                             in_h,
                                             in_w,
                                             in_c,
                                             weights,
                                             bias,
                                             kernel_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             pad_h_end,
                                             pad_w_end,
                                             out_channels,
                                             quant,
                                             clamp == QuantInteger::QuantClamp::ReLU ||
                                                 clamp == QuantInteger::QuantClamp::ReLU6,
                                             output))
        {
            wrote = true;
        }
#endif
        if (!wrote)
        {
            QuantTrace::RecordConv2dReference();
            const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
                in_h, kernel_size, stride, pad_h, pad_h_end);
            const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
                in_w, kernel_size, stride, pad_w, pad_w_end);
            const uint32_t kernel_area =
                static_cast<uint32_t>(kernel_size) * static_cast<uint32_t>(kernel_size);
            const uint32_t filter_elems = kernel_area * in_c;

            int32_t multiplier = 0;
            int32_t shift = 0;
            const bool per_channel =
                quant.weight_channel_scales != nullptr &&
                quant.num_weight_channel_scales == static_cast<uint32_t>(out_channels);
            if (!per_channel &&
                !QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                        quant.weight_scale,
                                                        quant.output_scale,
                                                        &multiplier,
                                                        &shift))
            {
                return;
            }

            for (size_t oh = 0; oh < out_h; ++oh)
            {
                for (size_t ow = 0; ow < out_w; ++ow)
                {
                    const uint32_t out_spatial_base =
                        (oh * out_w + ow) * static_cast<uint32_t>(out_channels);
                    for (int oc = 0; oc < out_channels; ++oc)
                    {
                        int32_t acc = bias[oc];
                        const int8_t* filter = weights + static_cast<uint32_t>(oc) * filter_elems;

                        for (int kh = 0; kh < kernel_size; ++kh)
                        {
                            const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                            if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                                continue;

                            for (int kw = 0; kw < kernel_size; ++kw)
                            {
                                const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                                if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                    continue;

                                const int8_t* in_ptr =
                                    input + (static_cast<uint32_t>(ih) * in_w +
                                             static_cast<uint32_t>(iw)) *
                                                in_c;
                                const int8_t* wt_ptr =
                                    filter + (static_cast<uint32_t>(kh) *
                                                  static_cast<uint32_t>(kernel_size) +
                                              static_cast<uint32_t>(kw)) *
                                                 in_c;

                                for (size_t ic = 0; ic < in_c; ++ic)
                                {
                                    const int32_t in_val =
                                        static_cast<int32_t>(in_ptr[ic]) - quant.input_zero_point;
                                    const int32_t wt_val =
                                        static_cast<int32_t>(wt_ptr[ic]) - quant.weight_zero_point;
                                    acc += in_val * wt_val;
                                }
                            }
                        }

                        int32_t m = multiplier;
                        int32_t s = shift;
                        if (per_channel)
                        {
                            if (!QuantInteger::EffectiveScaleMultiplier(
                                    quant.input_scale,
                                    quant.weight_channel_scales[oc],
                                    quant.output_scale,
                                    &m,
                                    &s))
                            {
                                return;
                            }
                        }
                        output[out_spatial_base + static_cast<uint32_t>(oc)] =
                            QuantInteger::RequantizeAccToInt8(acc,
                                                              m,
                                                              s,
                                                              quant.output_zero_point,
                                                              clamp,
                                                              quant.output_scale);
                    }
                }
            }
        }

        if (residual && residual->data)
        {
            const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
                in_h, kernel_size, stride, pad_h, pad_h_end);
            const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
                in_w, kernel_size, stride, pad_w, pad_w_end);
            const uint32_t count = out_h * out_w * static_cast<uint32_t>(out_channels);
            ElementwiseAddS8(output,
                             residual->data,
                             count,
                             quant.output_scale,
                             quant.output_zero_point,
                             residual->scale,
                             residual->zero_point,
                             quant.output_scale,
                             quant.output_zero_point,
                             output);
        }
    }

    void DepthwiseConv2dNhwcQuant(const int8_t* input,
                                  uint32_t in_h,
                                  uint32_t in_w,
                                  uint32_t channels,
                                  const int8_t* weights,
                                  const int32_t* bias,
                                  int kernel_h,
                                  int kernel_w,
                                  int stride,
                                  int pad_h,
                                  int pad_w,
                                  int pad_h_end,
                                  int pad_w_end,
                                  const NkFormat::MlpLayerQuantDesc& quant,
                                  QuantInteger::QuantClamp clamp,
                                  int8_t* output)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryDepthwiseConv2dNhwcQuant(input,
                                                      in_h,
                                                      in_w,
                                                      channels,
                                                      weights,
                                                      bias,
                                                      kernel_h,
                                                      kernel_w,
                                                      stride,
                                                      pad_h,
                                                      pad_w,
                                                      pad_h_end,
                                                      pad_w_end,
                                                      quant,
                                                      clamp == QuantInteger::QuantClamp::ReLU ||
                                                          clamp == QuantInteger::QuantClamp::ReLU6,
                                                      output))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryDepthwiseConv2dNhwcQuant(input,
                                                      in_h,
                                                      in_w,
                                                      channels,
                                                      weights,
                                                      bias,
                                                      kernel_h,
                                                      kernel_w,
                                                      stride,
                                                      pad_h,
                                                      pad_w,
                                                      pad_h_end,
                                                      pad_w_end,
                                                      quant,
                                                      clamp == QuantInteger::QuantClamp::ReLU ||
                                                          clamp == QuantInteger::QuantClamp::ReLU6,
                                                      output))
        {
            return;
        }
#endif
        QuantTrace::RecordConv2dReference();
        const uint32_t out_h =
            nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
        const uint32_t out_w =
            nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);
        const uint32_t kernel_area =
            static_cast<uint32_t>(kernel_h) * static_cast<uint32_t>(kernel_w);

        int32_t multiplier = 0;
        int32_t shift = 0;
        const bool per_channel =
            quant.weight_channel_scales != nullptr &&
            quant.num_weight_channel_scales == channels;
        if (!per_channel &&
            !QuantInteger::EffectiveScaleMultiplier(quant.input_scale,
                                                    quant.weight_scale,
                                                    quant.output_scale,
                                                    &multiplier,
                                                    &shift))
        {
            return;
        }

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
                for (uint32_t c = 0; c < channels; ++c)
                {
                    int32_t acc = bias[c];
                    const int8_t* filter = weights + c * kernel_area;

                    for (int kh = 0; kh < kernel_h; ++kh)
                    {
                        const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                        if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                            continue;

                        for (int kw = 0; kw < kernel_w; ++kw)
                        {
                            const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                            if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                continue;

                            const int8_t in_val_q =
                                input[(static_cast<uint32_t>(ih) * in_w +
                                       static_cast<uint32_t>(iw)) *
                                          channels +
                                      c];
                            const int8_t wt_val_q =
                                filter[static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_w) +
                                       static_cast<uint32_t>(kw)];
                            const int32_t in_val =
                                static_cast<int32_t>(in_val_q) - quant.input_zero_point;
                            const int32_t wt_val =
                                static_cast<int32_t>(wt_val_q) - quant.weight_zero_point;
                            acc += in_val * wt_val;
                        }
                    }

                    int32_t m = multiplier;
                    int32_t s = shift;
                    if (per_channel)
                    {
                        if (!QuantInteger::EffectiveScaleMultiplier(
                                quant.input_scale,
                                quant.weight_channel_scales[c],
                                quant.output_scale,
                                &m,
                                &s))
                        {
                            return;
                        }
                    }
                    output[out_spatial_base + c] = QuantInteger::RequantizeAccToInt8(
                        acc, m, s, quant.output_zero_point, clamp, quant.output_scale);
                }
            }
        }
    }

    void MaxPool2dNhwcQuant(const int8_t* input,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            int pool_h,
                            int pool_w,
                            int stride,
                            int pad_h,
                            int pad_w,
                            int pad_h_end,
                            int pad_w_end,
                            int8_t* output)
    {
#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
        if (XnnpackQuant::TryMaxPool2dNhwcQuant(input,
                                                in_h,
                                                in_w,
                                                in_c,
                                                pool_h,
                                                pool_w,
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_h_end,
                                                pad_w_end,
                                                output))
        {
            return;
        }
#endif
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryMaxPool2dNhwcQuant(input,
                                                in_h,
                                                in_w,
                                                in_c,
                                                pool_h,
                                                pool_w,
                                                stride,
                                                pad_h,
                                                pad_w,
                                                pad_h_end,
                                                pad_w_end,
                                                output))
        {
            return;
        }
#endif
        QuantTrace::RecordPoolReference();
        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool_h, stride, pad_h, pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool_w, stride, pad_w, pad_w_end);

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                for (size_t c = 0; c < in_c; ++c)
                {
                    int8_t max_val = -128;
                    bool found = false;

                    for (int kh = 0; kh < pool_h; ++kh)
                    {
                        const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                        if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                            continue;

                        for (int kw = 0; kw < pool_w; ++kw)
                        {
                            const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                            if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                continue;

                            const int8_t value = input[(static_cast<uint32_t>(ih) * in_w +
                                                        static_cast<uint32_t>(iw)) *
                                                           in_c +
                                                       c];
                            if (!found || value > max_val)
                            {
                                max_val = value;
                                found = true;
                            }
                        }
                    }

                    output[(oh * out_w + ow) * in_c + c] = found ? max_val : static_cast<int8_t>(0);
                }
            }
        }
    }

    void ElementwiseAddS8(const int8_t* input1,
                          const int8_t* input2,
                          uint32_t count,
                          float input1_scale,
                          int32_t input1_zero_point,
                          float input2_scale,
                          int32_t input2_zero_point,
                          float output_scale,
                          int32_t output_zero_point,
                          int8_t* output)
    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryElementwiseAddS8(input1,
                                              input2,
                                              count,
                                              input1_scale,
                                              input1_zero_point,
                                              input2_scale,
                                              input2_zero_point,
                                              output_scale,
                                              output_zero_point,
                                              output))
        {
            return;
        }
#endif
        // Integer path matching TFLite / CMSIS-NN elementwise_add_s8:
        //   twice_max = 2 * max(s1, s2)
        //   in_mult  = s_i / twice_max
        //   out_mult = twice_max / ((1 << left_shift) * s_out)
        constexpr int32_t kLeftShift = 20;
        const double twice_max_input_scale =
            2.0 * std::max(static_cast<double>(input1_scale), static_cast<double>(input2_scale));
        if (twice_max_input_scale <= 0.0 || output_scale <= 0.0f)
            return;

        int32_t input1_mult = 0;
        int32_t input1_shift = 0;
        int32_t input2_mult = 0;
        int32_t input2_shift = 0;
        int32_t output_mult = 0;
        int32_t output_shift = 0;
        QuantInteger::QuantizeMultiplier(static_cast<double>(input1_scale) / twice_max_input_scale,
                                         &input1_mult,
                                         &input1_shift);
        QuantInteger::QuantizeMultiplier(static_cast<double>(input2_scale) / twice_max_input_scale,
                                         &input2_mult,
                                         &input2_shift);
        QuantInteger::QuantizeMultiplier(
            twice_max_input_scale / ((1LL << kLeftShift) * static_cast<double>(output_scale)),
            &output_mult,
            &output_shift);

        for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t a = (static_cast<int32_t>(input1[i]) - input1_zero_point) << kLeftShift;
            const int32_t b = (static_cast<int32_t>(input2[i]) - input2_zero_point) << kLeftShift;
            const int32_t scaled_a =
                QuantInteger::MultiplyByQuantizedMultiplier(a, input1_mult, input1_shift);
            const int32_t scaled_b =
                QuantInteger::MultiplyByQuantizedMultiplier(b, input2_mult, input2_shift);
            const int32_t raw = QuantInteger::MultiplyByQuantizedMultiplier(
                scaled_a + scaled_b, output_mult, output_shift);
            const int32_t q = std::clamp(raw + output_zero_point, int32_t{-128}, int32_t{127});
            output[i] = static_cast<int8_t>(q);
        }
    }

    void AvgPool2dNhwcQuant(const int8_t* input,
                            uint32_t in_h,
                            uint32_t in_w,
                            uint32_t in_c,
                            int pool_h,
                            int pool_w,
                            int stride,
                            int pad_h,
                            int pad_w,
                            int pad_h_end,
                            int pad_w_end,
                            float input_scale,
                            int32_t input_zero_point,
                            float output_scale,
                            int32_t output_zero_point,
                            int8_t* output)
    {
        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, pool_h, stride, pad_h, pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, pool_w, stride, pad_w, pad_w_end);

        int32_t multiplier = 0;
        int32_t shift = 0;
        const bool same_scale =
            input_scale == output_scale && input_zero_point == output_zero_point;
        if (!same_scale)
        {
            if (input_scale <= 0.0f || output_scale <= 0.0f)
                return;
            QuantInteger::QuantizeMultiplier(static_cast<double>(input_scale) /
                                                 static_cast<double>(output_scale),
                                             &multiplier,
                                             &shift);
        }

        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                for (uint32_t c = 0; c < in_c; ++c)
                {
                    int32_t sum = 0;
                    uint32_t count = 0;

                    for (int kh = 0; kh < pool_h; ++kh)
                    {
                        const int32_t ih = static_cast<int32_t>(oh) * stride + kh - pad_h;
                        if (ih < 0 || ih >= static_cast<int32_t>(in_h))
                            continue;

                        for (int kw = 0; kw < pool_w; ++kw)
                        {
                            const int32_t iw = static_cast<int32_t>(ow) * stride + kw - pad_w;
                            if (iw < 0 || iw >= static_cast<int32_t>(in_w))
                                continue;

                            const int8_t value = input[(static_cast<uint32_t>(ih) * in_w +
                                                        static_cast<uint32_t>(iw)) *
                                                           in_c +
                                                       c];
                            sum += static_cast<int32_t>(value) - input_zero_point;
                            ++count;
                        }
                    }

                    int32_t avg = 0;
                    if (count > 0)
                    {
                        // Rounding divide in int32 (same sign as TFLite average_pool).
                        const int32_t abs_sum = sum >= 0 ? sum : -sum;
                        int32_t q = (abs_sum + static_cast<int32_t>(count / 2)) /
                                    static_cast<int32_t>(count);
                        avg = sum >= 0 ? q : -q;
                    }

                    int32_t out_q;
                    if (same_scale)
                        out_q = avg + output_zero_point;
                    else
                        out_q = QuantInteger::MultiplyByQuantizedMultiplier(avg, multiplier, shift) +
                                output_zero_point;
                    out_q = std::clamp(out_q, int32_t{-128}, int32_t{127});
                    output[(oh * out_w + ow) * in_c + c] = static_cast<int8_t>(out_q);
                }
            }
        }
    }

    void FlattenNhwcInt8(const int8_t* input, uint32_t num_elements, int8_t* output)
    {
        CmsisQuantUtil::CopyInt8(input, output, num_elements);
    }
}
