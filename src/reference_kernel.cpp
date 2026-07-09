#include "reference_kernel.hpp"
#include "cmsis_dsp_util.hpp"
#include "conv_dispatch.hpp"
#include "conv_depthwise_kernel.hpp"
#include "kernel_activation.hpp"
#include "netkit_loop_unroll.hpp"
#include "tensor_access.hpp"
#include <cmath>
#include <cfloat>

namespace
{
    float BiasAt(const Tensor& bias, uint32_t out_channel)
    {
        if (!bias.data)
            return 0.0f;

        const float* bias_data = static_cast<const float*>(bias.data);
        if (bias.rank == 1)
            return bias_data[out_channel * bias.stride[0]];

        return bias_data[out_channel * bias.stride[1]];
    }

    float ConvOutputValue(float sum, NetkitKernelActivation fuse_activation)
    {
        return ApplyKernelActivation(sum, fuse_activation);
    }

    bool fc_tensors_dense_row_major(const Tensor& input,
                                    const Tensor& weights,
                                    const Tensor& bias,
                                    const Tensor& output)
    {
        if (input.rank != 2 || weights.rank != 2 || output.rank != 2 || bias.rank != 2 || !bias.data)
            return false;

        const uint32_t batch = input.shape[0];
        const uint32_t in_features = input.shape[1];
        const uint32_t out_features = weights.shape[0];

        if (weights.shape[1] != in_features || output.shape[0] != batch || output.shape[1] != out_features ||
            bias.shape[0] != 1 || bias.shape[1] != out_features)
            return false;

        return input.stride[0] == in_features && input.stride[1] == 1 && weights.stride[0] == in_features &&
               weights.stride[1] == 1 && output.stride[0] == out_features && output.stride[1] == 1 &&
               bias.stride[0] == out_features && bias.stride[1] == 1;
    }

    bool fully_connected_dense_with_bias(const float* in,
                                         const float* wt,
                                         const float* bias,
                                         float* out,
                                         uint32_t batch,
                                         uint32_t in_features,
                                         uint32_t out_features,
                                         NetkitKernelActivation fuse_activation)
    {
        const bool fuse_in_kernel = kernel_activation_is_fused(fuse_activation);

        for (size_t b = 0; b < batch; ++b)
        {
            const float* in_row = in + b * in_features;
            float* out_row = out + b * out_features;

            for (size_t oc = 0; oc < out_features; ++oc)
            {
                // DotProductF32 is header-inline: the reference build inlines the 4-accumulator
                // dot_contiguous here (no cross-TU call, no LTO needed); CMSIS uses arm_dot_prod_f32.
                const float sum =
                    CmsisDspUtil::DotProductF32(in_row, wt + oc * in_features, in_features);
                const float value = ApplyKernelActivation(sum + bias[oc], fuse_activation);
                out_row[oc] = value;
            }
        }

        return fuse_in_kernel;
    }

    bool conv_padding_zero(int pad_h, int pad_w, int pad_h_end, int pad_w_end)
    {
        return pad_h == 0 && pad_w == 0 && pad_h_end == 0 && pad_w_end == 0;
    }

    void MaxPool2dForward2x2S2P0(const float* in,
                                 float* out,
                                 uint32_t in_w,
                                 uint32_t channels,
                                 uint32_t out_h,
                                 uint32_t out_w)
    {
        for (size_t oh = 0; oh < out_h; ++oh)
        {
            const uint32_t ih0 = oh * 2u;
            const uint32_t row0 = ih0 * in_w;
            const uint32_t row1 = (ih0 + 1u) * in_w;

            for (size_t ow = 0; ow < out_w; ++ow)
            {
                const uint32_t iw0 = ow * 2u;
                const uint32_t out_spatial_base = (oh * out_w + ow) * channels;
                const uint32_t base0 = (row0 + iw0) * channels;
                const uint32_t base1 = base0 + channels;
                const uint32_t base2 = (row1 + iw0) * channels;
                const uint32_t base3 = base2 + channels;

                for (size_t c = 0; c < channels; ++c)
                {
                    float max_val = in[base0 + c];
                    const float v1 = in[base1 + c];
                    if (v1 > max_val)
                        max_val = v1;
                    const float v2 = in[base2 + c];
                    if (v2 > max_val)
                        max_val = v2;
                    const float v3 = in[base3 + c];
                    if (v3 > max_val)
                        max_val = v3;
                    out[out_spatial_base + c] = max_val;
                }
            }
        }
    }
}

void ReferenceKernel::MulImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    CmsisDspUtil::MulF32(static_cast<const float*>(a.data),
                         static_cast<const float*>(b.data),
                         static_cast<float*>(c.data),
                         a.num_elements);
}

void ReferenceKernel::MulScalarImpl(const Tensor& a, float scalar, Tensor& c)
{
    CmsisDspUtil::MulScalarF32(static_cast<const float*>(a.data),
                               scalar,
                               static_cast<float*>(c.data),
                               a.num_elements);
}

void ReferenceKernel::MatAddImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    const uint32_t rows = a.shape[0];
    const uint32_t cols = a.shape[1];

    if (a.stride[0] == cols && a.stride[1] == 1 && b.stride[0] == cols && b.stride[1] == 1 &&
        c.stride[0] == cols && c.stride[1] == 1)
    {
        CmsisDspUtil::AddF32(a_data, b_data, c_data, rows * cols);
        return;
    }

    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            const uint32_t a_idx = i * a.stride[0] + j * a.stride[1];
            const uint32_t b_idx = i * b.stride[0] + j * b.stride[1];
            const uint32_t c_idx = i * c.stride[0] + j * c.stride[1];
            c_data[c_idx] = a_data[a_idx] + b_data[b_idx];
        }
    }
}

void ReferenceKernel::MatAddNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    CmsisDspUtil::AddF32(static_cast<const float*>(a.data),
                         static_cast<const float*>(b.data),
                         static_cast<float*>(c.data),
                         a.num_elements);
}

void ReferenceKernel::MatMulImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    const uint32_t m = a.shape[0];
    const uint32_t k = a.shape[1];
    const uint32_t n = b.shape[1];

    for (size_t i = 0; i < m; i++)
    {
        const float* a_row = a_data + i * a.stride[0];
        for (size_t j = 0; j < n; j++)
        {
            const float sum = NetkitLoopUnroll::dot_strided_b_offset(
                a_row, a.stride[1], b_data, b.stride[0], j * b.stride[1], k);
            c_data[i * c.stride[0] + j * c.stride[1]] = sum;
        }
    }
}

void ReferenceKernel::MulNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    CmsisDspUtil::MulF32(static_cast<const float*>(a.data),
                         static_cast<const float*>(b.data),
                         static_cast<float*>(c.data),
                         a.num_elements);
}

void ReferenceKernel::FullyConnectedImpl(const Tensor& input, const Tensor& kernel, Tensor& output)
{
    const float* in = static_cast<const float*>(input.data);
    const float* wt = static_cast<const float*>(kernel.data);
    float* out = static_cast<float*>(output.data);

    const uint32_t batch = input.shape[0];
    const uint32_t in_features = input.shape[1];
    const uint32_t out_features = kernel.shape[0];

    for (size_t b = 0; b < batch; ++b)
    {
        const float* in_row = in + b * input.stride[0];
        for (size_t oc = 0; oc < out_features; ++oc)
        {
            const float* wt_row = wt + oc * kernel.stride[0];
            const float sum =
                NetkitLoopUnroll::dot_strided(in_row, input.stride[1], wt_row, kernel.stride[1], in_features);
            out[b * output.stride[0] + oc * output.stride[1]] = sum;
        }
    }
}

bool ReferenceKernel::FullyConnectedWithBiasImpl(const Tensor& input,
                                                 const Tensor& weights,
                                                 const Tensor& bias,
                                                 NetkitKernelActivation fuse_activation,
                                                 Tensor& output)
{
    if (fc_tensors_dense_row_major(input, weights, bias, output))
    {
        return fully_connected_dense_with_bias(static_cast<const float*>(input.data),
                                               static_cast<const float*>(weights.data),
                                               static_cast<const float*>(bias.data),
                                               static_cast<float*>(output.data),
                                               input.shape[0],
                                               input.shape[1],
                                               weights.shape[0],
                                               fuse_activation);
    }

    const float* in = static_cast<const float*>(input.data);
    const float* wt = static_cast<const float*>(weights.data);
    float* out = static_cast<float*>(output.data);

    const uint32_t batch = input.shape[0];
    const uint32_t in_features = input.shape[1];
    const uint32_t out_features = weights.shape[0];

    for (size_t b = 0; b < batch; ++b)
    {
        const float* in_row = in + b * input.stride[0];
        for (size_t oc = 0; oc < out_features; ++oc)
        {
            const float* wt_row = wt + oc * weights.stride[0];
            const float sum =
                NetkitLoopUnroll::dot_strided(in_row, input.stride[1], wt_row, weights.stride[1], in_features);
            const float value = ApplyKernelActivation(sum + BiasAt(bias, oc), fuse_activation);
            out[b * output.stride[0] + oc * output.stride[1]] = value;
        }
    }

    return kernel_activation_is_fused(fuse_activation);
}

void ReferenceKernel::ReLUImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) {
        c_data[i] = (a_data[i] > 0.0f) ? a_data[i] : 0.0f;
    });
}

void ReferenceKernel::SigmoidImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) {
        c_data[i] = 1.0f / (1.0f + std::expf(-a_data[i]));
    });
}

void ReferenceKernel::TanhImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) { c_data[i] = std::tanhf(a_data[i]); });
}

void ReferenceKernel::LeakyReLUImpl(const Tensor& a, Tensor& c, float alpha)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) {
        c_data[i] = (a_data[i] > 0.0f) ? a_data[i] : alpha * a_data[i];
    });
}

void ReferenceKernel::ReLU6Impl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) {
        const float x = a_data[i];
        if (x < 0.0f)
            c_data[i] = 0.0f;
        else if (x > 6.0f)
            c_data[i] = 6.0f;
        else
            c_data[i] = x;
    });
}

void ReferenceKernel::SoftmaxImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    const uint32_t n = a.num_elements;
    float max_val = a_data[0];
    for (size_t i = 1; i < n; i++)
    {
        if (a_data[i] > max_val)
            max_val = a_data[i];
    }

    float sum = 0.0f;
    NetkitLoopUnroll::for_count(n, [&](uint32_t i) {
        const float e = std::expf(a_data[i] - max_val);
        c_data[i] = e;
        sum += e;
    });

    CmsisDspUtil::ScaleF32(c_data, 1.0f / sum, n);
}

namespace
{
    constexpr float kGeluCoef = 0.044715f;
    constexpr float kSqrt2OverPi = 0.7978845608f;
}

void ReferenceKernel::GeluImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    NetkitLoopUnroll::for_count(a.num_elements, [&](uint32_t i) {
        const float x = a_data[i];
        const float inner = kSqrt2OverPi * (x + kGeluCoef * x * x * x);
        c_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    });
}

void ReferenceKernel::Grn2dForwardImpl(const Tensor& input,
                                       const float* gamma,
                                       const float* beta,
                                       int channels,
                                       float eps,
                                       float* channel_norm_scratch,
                                       Tensor& output)
{
    if (!gamma || !beta || !channel_norm_scratch || input.rank != 3 || output.rank != 3)
        return;

    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);
    const uint32_t height = input.shape[0];
    const uint32_t width = input.shape[1];
    const uint32_t channel_count = static_cast<uint32_t>(channels);
    const uint32_t spatial = height * width;

    if (input.shape[2] != channel_count || output.shape[2] != channel_count)
        return;

    for (size_t c = 0; c < channel_count; ++c)
    {
        double sum_sq = 0.0;
        NetkitLoopUnroll::for_count(spatial, [&](uint32_t i) {
            const float value = in[i * channel_count + c];
            sum_sq += static_cast<double>(value) * value;
        });
        channel_norm_scratch[c] = std::sqrt(static_cast<float>(sum_sq));
    }

    float mean_norm = 0.0f;
    for (size_t c = 0; c < channel_count; ++c)
        mean_norm += channel_norm_scratch[c];
    mean_norm /= static_cast<float>(channel_count);

    const float denom = mean_norm + eps;
    for (size_t i = 0; i < spatial; ++i)
    {
        for (size_t c = 0; c < channel_count; ++c)
        {
            const float nx = channel_norm_scratch[c] / denom;
            const float x = in[i * channel_count + c];
            out[i * channel_count + c] = gamma[c] * (x * nx) + beta[c] + x;
        }
    }
}

bool ReferenceKernel::Conv2dForwardImpl(const Tensor& input,
                                        float* weights,
                                        float* bias,
                                        int kernel_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        int pad_h_end,
                                        int pad_w_end,
                                        int in_channels,
                                        int out_channels,
                                        NetkitKernelActivation fuse_activation,
                                        Tensor& output,
                                        const float* weights_hwio)
{
    return Conv2dDispatchForward(input,
                                 weights,
                                 bias,
                                 kernel_size,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 in_channels,
                                 out_channels,
                                 fuse_activation,
                                 output,
                                 weights_hwio);
}

bool ReferenceKernel::DepthwiseConv2dForwardImpl(const Tensor& input,
                                                 float* weights,
                                                 float* bias,
                                                 int kernel_h,
                                                 int kernel_w,
                                                 int stride,
                                                 int pad_h,
                                                 int pad_w,
                                                 int pad_h_end,
                                                 int pad_w_end,
                                                 int channels,
                                                 NetkitKernelActivation fuse_activation,
                                                 Tensor& output)
{
    return ConvDepthwiseForward(input,
                              weights,
                              bias,
                              kernel_h,
                              kernel_w,
                              stride,
                              pad_h,
                              pad_w,
                              pad_h_end,
                              pad_w_end,
                              channels,
                              fuse_activation,
                              output);
}

bool ReferenceKernel::MaxPool2dForwardImpl(const Tensor& input,
                                           int pool_h,
                                           int pool_w,
                                           int stride,
                                           int pad_h,
                                           int pad_w,
                                           int pad_h_end,
                                           int pad_w_end,
                                           NetkitKernelActivation fuse_activation,
                                           Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t in_w = input.shape[1];
    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    if (pool_h == 2 && pool_w == 2 && stride == 2 && pad_h == 0 && pad_w == 0 && pad_h_end == 0 &&
        pad_w_end == 0)
    {
        MaxPool2dForward2x2S2P0(in, out, in_w, channels, out_h, out_w);
        if (kernel_activation_is_fused(fuse_activation))
        {
            const uint32_t n = out_h * out_w * channels;
            for (uint32_t i = 0; i < n; ++i)
                out[i] = ApplyKernelActivation(out[i], fuse_activation);
        }
        return kernel_activation_is_fused(fuse_activation);
    }

    const uint32_t in_h = input.shape[0];

    for (size_t oh = 0; oh < out_h; ++oh)
    {
        for (size_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;

            for (size_t c = 0; c < channels; ++c)
            {
                float max_val = -FLT_MAX;
                for (int kh = 0; kh < pool_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || static_cast<uint32_t>(ih) >= in_h)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w;

                    for (int kw = 0; kw < pool_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw >= 0 && static_cast<uint32_t>(iw) < in_w)
                        {
                            const float v =
                                in[(in_row + static_cast<uint32_t>(iw)) * channels + c];
                            if (v > max_val)
                                max_val = v;
                        }
                    }
                }

                out[out_spatial_base + c] = ApplyKernelActivation(max_val, fuse_activation);
            }
        }
    }
    return kernel_activation_is_fused(fuse_activation);
}

void ReferenceKernel::AvgPool2dForwardImpl(const Tensor& input,
                                           int pool_h,
                                           int pool_w,
                                           int stride,
                                           int pad_h,
                                           int pad_w,
                                           int /*pad_h_end*/,
                                           int /*pad_w_end*/,
                                           Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (size_t oh = 0; oh < out_h; ++oh)
    {
        for (size_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;

            for (size_t c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                uint32_t count = 0;
                for (int kh = 0; kh < pool_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || static_cast<uint32_t>(ih) >= in_h)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w;

                    for (int kw = 0; kw < pool_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw >= 0 && static_cast<uint32_t>(iw) < in_w)
                        {
                            sum += in[(in_row + static_cast<uint32_t>(iw)) * channels + c];
                            ++count;
                        }
                    }
                }

                out[out_spatial_base + c] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
            }
        }
    }
}

void ReferenceKernel::BatchNorm2dForwardImpl(const Tensor& input,
                                             const float* scale,
                                             const float* bias,
                                             int channels,
                                             Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);
    const uint32_t channels_u = static_cast<uint32_t>(channels);

    NetkitLoopUnroll::for_count(input.num_elements, [&](uint32_t i) {
        const uint32_t c = i % channels_u;
        out[i] = in[i] * scale[c] + bias[c];
    });
}

void ReferenceKernel::LayerNorm2dForwardImpl(const Tensor& input,
                                             const float* weight,
                                             const float* bias,
                                             int channels,
                                             float eps,
                                             Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);
    const uint32_t height = input.shape[0];
    const uint32_t width = input.shape[1];
    const uint32_t channel_count = static_cast<uint32_t>(channels);

    for (size_t oh = 0; oh < height; ++oh)
    {
        for (size_t ow = 0; ow < width; ++ow)
        {
            const float* pixel_in = in + (oh * width + ow) * channel_count;
            float* pixel_out = out + (oh * width + ow) * channel_count;

            float mean = 0.0f;
            for (size_t c = 0; c < channel_count; ++c)
                mean += pixel_in[c];
            mean /= static_cast<float>(channel_count);

            float variance = 0.0f;
            NetkitLoopUnroll::for_count(channel_count, [&](uint32_t c) {
                const float delta = pixel_in[c] - mean;
                variance += delta * delta;
            });
            variance /= static_cast<float>(channel_count);
            const float inv_std = 1.0f / std::sqrt(variance + eps);

            NetkitLoopUnroll::for_count(channel_count, [&](uint32_t c) {
                pixel_out[c] = (pixel_in[c] - mean) * inv_std * weight[c] + bias[c];
            });
        }
    }
}
