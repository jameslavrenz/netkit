#include "convnextv2_block.hpp"

#include "active_kernel.hpp"
#include "tensor_access.hpp"

#include <cmath>
#include <cstring>

namespace
{
    constexpr float kGeluCoef = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;

    void GeluInPlace(float* data, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            const float x = data[i];
            const float inner = kSqrt2OverPi * (x + kGeluCoef * x * x * x);
            data[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    void LayerNormNhwcInPlace(float* data,
                              uint32_t height,
                              uint32_t width,
                              uint32_t channel_count,
                              const float* weight,
                              const float* bias,
                              float eps)
    {
        const uint32_t channels = channel_count;
        for (uint32_t oh = 0; oh < height; ++oh)
        {
            for (uint32_t ow = 0; ow < width; ++ow)
            {
                float* pixel = data + (oh * width + ow) * channels;
                float mean = 0.0f;
                for (uint32_t c = 0; c < channels; ++c)
                    mean += pixel[c];
                mean /= static_cast<float>(channels);

                float variance = 0.0f;
                for (uint32_t c = 0; c < channels; ++c)
                {
                    const float delta = pixel[c] - mean;
                    variance += delta * delta;
                }
                variance /= static_cast<float>(channels);

                const float inv_std = 1.0f / std::sqrtf(variance + eps);
                for (uint32_t c = 0; c < channels; ++c)
                    pixel[c] = (pixel[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }

    void GrnNhwcInPlace(float* data,
                        uint32_t height,
                        uint32_t width,
                        uint32_t channel_count,
                        const float* gamma,
                        const float* beta,
                        float eps,
                        float* channel_norms)
    {
        const uint32_t spatial = height * width;
        const uint32_t channels = channel_count;

        for (uint32_t c = 0; c < channels; ++c)
        {
            double sum_sq = 0.0;
            for (uint32_t i = 0; i < spatial; ++i)
            {
                const float value = data[i * channels + c];
                sum_sq += static_cast<double>(value) * value;
            }
            channel_norms[c] = std::sqrt(static_cast<float>(sum_sq));
        }

        float mean_norm = 0.0f;
        for (uint32_t c = 0; c < channels; ++c)
            mean_norm += channel_norms[c];
        mean_norm /= static_cast<float>(channels);

        const float denom = mean_norm + eps;
        for (uint32_t i = 0; i < spatial; ++i)
        {
            for (uint32_t c = 0; c < channels; ++c)
            {
                const float nx = channel_norms[c] / denom;
                const float x = data[i * channels + c];
                data[i * channels + c] = gamma[c] * (x * nx) + beta[c] + x;
            }
        }
    }

    void DensePixel(const float* input,
                    int in_features,
                    int out_features,
                    const float* weight,
                    const float* bias,
                    float* output)
    {
        for (int out = 0; out < out_features; ++out)
        {
            float sum = bias ? bias[out] : 0.0f;
            const float* row = weight + static_cast<std::size_t>(out) * in_features;
            for (int in = 0; in < in_features; ++in)
                sum += row[in] * input[in];
            output[out] = sum;
        }
    }
}

void ConvNeXtV2Block::forward(const Tensor& input, Tensor& output)
{
    const uint32_t height = input.shape[0];
    const uint32_t width = input.shape[1];
    const uint32_t channel_count = static_cast<uint32_t>(channels);
    const uint32_t expanded = channel_count * static_cast<uint32_t>(kMlpRatio);
    const uint32_t spatial = height * width;
    const uint32_t required_scratch = spatial * expanded + expanded;

    if (!scratch || scratch_elems < required_scratch)
        return;

    float* branch = tensor_data_f32(output);
    float* expanded_buf = scratch;
    float* grn_norms = scratch + spatial * expanded;

    Kernels::DepthwiseConv2dForward(input,
                                    dw_weights,
                                    dw_bias,
                                    kDwKernel,
                                    kDwKernel,
                                    1,
                                    kDwPad,
                                    kDwPad,
                                    channels,
                                    NetkitKernelActivation::None,
                                    output);

    LayerNormNhwcInPlace(branch, height, width, channel_count, ln_weight, ln_bias, eps);

    for (uint32_t i = 0; i < spatial; ++i)
    {
        const float* pixel_in = branch + i * channel_count;
        float* pixel_out = expanded_buf + i * expanded;
        DensePixel(pixel_in,
                   channels,
                   static_cast<int>(expanded),
                   pw1_weight,
                   pw1_bias,
                   pixel_out);
    }

    GeluInPlace(expanded_buf, spatial * expanded);
    GrnNhwcInPlace(expanded_buf,
                   height,
                   width,
                   expanded,
                   grn_gamma,
                   grn_beta,
                   eps,
                   grn_norms);

    const float* residual = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);
    for (uint32_t i = 0; i < spatial; ++i)
    {
        const float* pixel_in = expanded_buf + i * expanded;
        float* pixel_out = out + i * channel_count;
        DensePixel(pixel_in,
                   static_cast<int>(expanded),
                   channels,
                   pw2_weight,
                   pw2_bias,
                   pixel_out);
        for (uint32_t c = 0; c < channel_count; ++c)
            pixel_out[c] += residual[i * channel_count + c];
    }
}
