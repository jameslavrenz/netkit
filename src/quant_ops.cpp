#include "quant_ops.hpp"

#include "cmsis_nn_quant.hpp"
#include "cmsis_dsp_util.hpp"
#include "netkit_config.h"
#include "nk_op_detail.hpp"
#include "quant_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace QuantOps
{
    int8_t QuantizeFloat(float value, float scale, int32_t zero_point)
    {
        if (scale <= 0.0f)
            return static_cast<int8_t>(zero_point);

        const float inv_scale = 1.0f / scale;
        int32_t q = static_cast<int32_t>(std::lround(value * inv_scale)) + zero_point;
        q = std::clamp(q, int32_t{-128}, int32_t{127});
        return static_cast<int8_t>(q);
    }

    float DequantizeInt8(int8_t value, float scale, int32_t zero_point)
    {
        return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
    }

    void RequantizeToInt8(const float* values,
                          uint32_t count,
                          float output_scale,
                          int32_t output_zero_point,
                          int8_t* output)
    {
        for (size_t i = 0; i < count; ++i)
            output[i] = QuantizeFloat(values[i], output_scale, output_zero_point);
    }

    void FullyConnectedQuant(const int8_t* input,
                             uint32_t batch,
                             uint32_t in_features,
                             const int8_t* weights,
                             const int32_t* bias,
                             uint32_t out_features,
                             const NkFormat::MlpLayerQuantDesc& quant,
                             bool apply_relu,
                             int8_t* output_int8,
                             float* output_float)
    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryFullyConnectedQuant(input,
                                                 batch,
                                                 in_features,
                                                 weights,
                                                 bias,
                                                 out_features,
                                                 quant,
                                                 apply_relu,
                                                 output_int8,
                                                 output_float))
        {
            return;
        }
#endif
        QuantTrace::RecordFcReference();
        const float effective_scale = quant.input_scale * quant.weight_scale;

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

                float out_real = static_cast<float>(acc) * effective_scale;
                if (apply_relu)
                    out_real = std::max(0.0f, out_real);

                if (output_float != nullptr)
                    output_float[b * out_features + oc] = out_real;
                else if (output_int8 != nullptr)
                    output_int8[b * out_features + oc] =
                        QuantizeFloat(out_real, quant.output_scale, quant.output_zero_point);
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
                               bool input_is_float,
                               Tensor& output)
    {
        const uint32_t batch = input.shape[0];
        const uint32_t in_features = input.shape[1];
        const uint32_t out_features = weights.shape[0];

        const int8_t* input_i8 = nullptr;
        if (input_is_float)
        {
            const float* in_f = static_cast<const float*>(input.data);
            for (size_t i = 0; i < batch * in_features; ++i)
                quant_scratch[i] =
                    QuantizeFloat(in_f[i], quant.params.input_scale, quant.params.input_zero_point);
            input_i8 = quant_scratch;
        }
        else
        {
            input_i8 = static_cast<const int8_t*>(input.data);
        }

        const int8_t* weight_q = static_cast<const int8_t*>(weights.data);
        const int32_t* bias_q = static_cast<const int32_t*>(bias.data);

        int8_t* logits_i8 = quant_scratch;
        if (input_is_float)
            logits_i8 = quant_scratch + batch * in_features;

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
                                logits_i8,
                                nullptr);

            const float logit_scale =
                quant.params.output_scale > 0.0f ? quant.params.output_scale : 1.0f;

            if (output.type == DataType::Int8)
            {
                SoftmaxS8(logits_i8, out_features, logit_scale, static_cast<int8_t*>(output.data));
                return;
            }

            float* out_f = static_cast<float*>(output.data);
            int8_t* probs_i8 = logits_i8;
            if (input_is_float)
                probs_i8 = quant_scratch + batch * in_features + out_features;

            SoftmaxS8(logits_i8, out_features, logit_scale, probs_i8);
            DequantizeSoftmaxOutput(probs_i8, out_f, out_features);
            return;
        }

        if (output.type == DataType::Float32)
        {
            float* out_f = static_cast<float*>(output.data);
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
            if (CmsisNnQuant::TryFullyConnectedQuantToFloat(input_i8,
                                                            batch,
                                                            in_features,
                                                            weight_q,
                                                            bias_q,
                                                            out_features,
                                                            quant.params,
                                                            apply_relu,
                                                            logits_i8,
                                                            out_f))
            {
                return;
            }
#endif
            FullyConnectedQuant(input_i8,
                                batch,
                                in_features,
                                weight_q,
                                bias_q,
                                out_features,
                                quant.params,
                                apply_relu,
                                nullptr,
                                out_f);
            return;
        }

        int8_t* out_i8 = static_cast<int8_t*>(output.data);
        FullyConnectedQuant(input_i8,
                            batch,
                            in_features,
                            weight_q,
                            bias_q,
                            out_features,
                            quant.params,
                            apply_relu,
                            out_i8,
                            nullptr);
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
                         bool apply_relu,
                         int8_t* output)
    {
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
        if (CmsisNnQuant::TryConv2dNhwcQuant(input,
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
                                             apply_relu,
                                             output))
        {
            return;
        }
#endif
        QuantTrace::RecordConv2dReference();
        const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
            in_h, kernel_size, stride, pad_h, pad_h_end);
        const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
            in_w, kernel_size, stride, pad_w, pad_w_end);
        const uint32_t kernel_area = static_cast<uint32_t>(kernel_size) * static_cast<uint32_t>(kernel_size);
        const uint32_t filter_elems = kernel_area * in_c;
        const float effective_scale = quant.input_scale * quant.weight_scale;

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t out_spatial_base = (oh * out_w + ow) * static_cast<uint32_t>(out_channels);
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
                                input + (static_cast<uint32_t>(ih) * in_w + static_cast<uint32_t>(iw)) * in_c;
                            const int8_t* wt_ptr = filter + (static_cast<uint32_t>(kh) * static_cast<uint32_t>(kernel_size) +
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

                    float out_real = static_cast<float>(acc) * effective_scale;
                    if (apply_relu)
                        out_real = std::max(0.0f, out_real);
                    output[out_spatial_base + static_cast<uint32_t>(oc)] =
                        QuantizeFloat(out_real, quant.output_scale, quant.output_zero_point);
                }
            }
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
                                  bool apply_relu,
                                  int8_t* output)
    {
        QuantTrace::RecordConv2dReference();
        const uint32_t out_h =
            nk_op_detail::CalcOutputDimAsymmetric(in_h, kernel_h, stride, pad_h, pad_h_end);
        const uint32_t out_w =
            nk_op_detail::CalcOutputDimAsymmetric(in_w, kernel_w, stride, pad_w, pad_w_end);
        const uint32_t kernel_area =
            static_cast<uint32_t>(kernel_h) * static_cast<uint32_t>(kernel_w);
        const float effective_scale = quant.input_scale * quant.weight_scale;

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

                    float out_real = static_cast<float>(acc) * effective_scale;
                    if (apply_relu)
                        out_real = std::max(0.0f, out_real);
                    output[out_spatial_base + c] =
                        QuantizeFloat(out_real, quant.output_scale, quant.output_zero_point);
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
        for (uint32_t i = 0; i < count; ++i)
        {
            const float v1 = DequantizeInt8(input1[i], input1_scale, input1_zero_point);
            const float v2 = DequantizeInt8(input2[i], input2_scale, input2_zero_point);
            output[i] = QuantizeFloat(v1 + v2, output_scale, output_zero_point);
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

        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                for (uint32_t c = 0; c < in_c; ++c)
                {
                    float sum = 0.0f;
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
                            sum += DequantizeInt8(value, input_scale, input_zero_point);
                            ++count;
                        }
                    }

                    const float avg = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                    output[(oh * out_w + ow) * in_c + c] =
                        QuantizeFloat(avg, output_scale, output_zero_point);
                }
            }
        }
    }

    void FlattenNhwcInt8(const int8_t* input, uint32_t num_elements, int8_t* output)
    {
        CmsisQuantUtil::CopyInt8(input, output, num_elements);
    }
}
