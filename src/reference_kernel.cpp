#include "reference_kernel.hpp"
#include "netkit_loop_unroll.hpp"
#include "tensor_access.hpp"
#include <cmath>
#include <cfloat>

void ReferenceKernel::MulImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    NetkitLoopUnroll::mul_contiguous(static_cast<const float*>(a.data),
                                     static_cast<const float*>(b.data),
                                     static_cast<float*>(c.data),
                                     a.num_elements);
}

void ReferenceKernel::MulScalarImpl(const Tensor& a, float scalar, Tensor& c)
{
    NetkitLoopUnroll::mul_scalar_contiguous(static_cast<const float*>(a.data),
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
        NetkitLoopUnroll::add_contiguous(a_data, b_data, c_data, rows * cols);
        return;
    }

    for (uint32_t i = 0; i < rows; i++)
    {
        for (uint32_t j = 0; j < cols; j++)
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
    NetkitLoopUnroll::add_contiguous(static_cast<const float*>(a.data),
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

    for (uint32_t i = 0; i < m; i++)
    {
        const float* a_row = a_data + i * a.stride[0];
        for (uint32_t j = 0; j < n; j++)
        {
            const float sum = NetkitLoopUnroll::dot_strided_b_offset(
                a_row, a.stride[1], b_data, b.stride[0], j * b.stride[1], k);
            c_data[i * c.stride[0] + j * c.stride[1]] = sum;
        }
    }
}

void ReferenceKernel::MulNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    NetkitLoopUnroll::mul_contiguous(static_cast<const float*>(a.data),
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

    for (uint32_t b = 0; b < batch; ++b)
    {
        const float* in_row = in + b * input.stride[0];
        for (uint32_t oc = 0; oc < out_features; ++oc)
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
                                                 NetkitKernelActivation /*fuse_activation*/,
                                                 Tensor& output)
{
    FullyConnectedImpl(input, weights, output);
    MatAddImpl(output, bias, output);
    return false;
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
    for (uint32_t i = 1; i < n; i++)
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

    NetkitLoopUnroll::scale_contiguous(c_data, 1.0f / sum, n);
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

    for (uint32_t c = 0; c < channel_count; ++c)
    {
        double sum_sq = 0.0;
        NetkitLoopUnroll::for_count(spatial, [&](uint32_t i) {
            const float value = in[i * channel_count + c];
            sum_sq += static_cast<double>(value) * value;
        });
        channel_norm_scratch[c] = std::sqrt(static_cast<float>(sum_sq));
    }

    float mean_norm = 0.0f;
    for (uint32_t c = 0; c < channel_count; ++c)
        mean_norm += channel_norm_scratch[c];
    mean_norm /= static_cast<float>(channel_count);

    const float denom = mean_norm + eps;
    for (uint32_t i = 0; i < spatial; ++i)
    {
        for (uint32_t c = 0; c < channel_count; ++c)
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
                                        NetkitKernelActivation /*fuse_activation*/,
                                        Tensor& output)
{
    (void)pad_h_end;
    (void)pad_w_end;

    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];
    const int in_h = static_cast<int>(input.shape[0]);
    const int in_w = static_cast<int>(input.shape[1]);
    const uint32_t in_w_u = input.shape[1];
    const uint32_t in_ch = static_cast<uint32_t>(in_channels);
    const uint32_t out_ch = static_cast<uint32_t>(out_channels);

    for (uint32_t oh = 0; oh < out_h; oh++)
    {
        for (uint32_t ow = 0; ow < out_w; ow++)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * out_ch;

            for (int oc = 0; oc < out_channels; oc++)
            {
                float sum = bias ? bias[oc] : 0.0f;

                for (int kh = 0; kh < kernel_size; kh++)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || ih >= in_h)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w_u;

                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw < 0 || iw >= in_w)
                            continue;

                        const uint32_t in_base = (in_row + static_cast<uint32_t>(iw)) * input.shape[2];
                        const uint32_t w_base =
                            ((static_cast<uint32_t>(oc) * static_cast<uint32_t>(kernel_size) +
                              static_cast<uint32_t>(kh)) *
                                 static_cast<uint32_t>(kernel_size) +
                             static_cast<uint32_t>(kw)) *
                            in_ch;

                        for (uint32_t ic = 0; ic < in_ch; ++ic)
                            sum += in[in_base + ic] * weights[w_base + ic];
                    }
                }

                out[out_spatial_base + static_cast<uint32_t>(oc)] = sum;
            }
        }
    }

    return false;
}

bool ReferenceKernel::DepthwiseConv2dForwardImpl(const Tensor& input,
                                                 float* weights,
                                                 float* bias,
                                                 int kernel_h,
                                                 int kernel_w,
                                                 int stride,
                                                 int pad_h,
                                                 int pad_w,
                                                 int /*pad_h_end*/,
                                                 int /*pad_w_end*/,
                                                 int channels,
                                                 NetkitKernelActivation /*fuse_activation*/,
                                                 Tensor& output)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];
    const int in_h = static_cast<int>(input.shape[0]);
    const int in_w = static_cast<int>(input.shape[1]);
    const uint32_t in_w_u = input.shape[1];
    const uint32_t ch_u = static_cast<uint32_t>(channels);

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * ch_u;

            for (int c = 0; c < channels; ++c)
            {
                float sum = bias ? bias[c] : 0.0f;

                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                    if (ih < 0 || ih >= in_h)
                        continue;

                    const uint32_t in_row = static_cast<uint32_t>(ih) * in_w_u;

                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (iw < 0 || iw >= in_w)
                            continue;

                        const uint32_t in_idx =
                            (in_row + static_cast<uint32_t>(iw)) * ch_u + static_cast<uint32_t>(c);
                        const uint32_t w_idx =
                            (static_cast<uint32_t>(c) * static_cast<uint32_t>(kernel_h) +
                             static_cast<uint32_t>(kh)) *
                                static_cast<uint32_t>(kernel_w) +
                            static_cast<uint32_t>(kw);
                        sum += in[in_idx] * weights[w_idx];
                    }
                }

                out[out_spatial_base + static_cast<uint32_t>(c)] = sum;
            }
        }
    }

    return false;
}

void ReferenceKernel::MaxPool2dForwardImpl(const Tensor& input,
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

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;

            for (uint32_t c = 0; c < channels; ++c)
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

                out[out_spatial_base + c] = max_val;
            }
        }
    }
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

    for (uint32_t oh = 0; oh < out_h; ++oh)
    {
        for (uint32_t ow = 0; ow < out_w; ++ow)
        {
            const uint32_t out_spatial_base = (oh * out_w + ow) * channels;

            for (uint32_t c = 0; c < channels; ++c)
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

    for (uint32_t oh = 0; oh < height; ++oh)
    {
        for (uint32_t ow = 0; ow < width; ++ow)
        {
            const float* pixel_in = in + (oh * width + ow) * channel_count;
            float* pixel_out = out + (oh * width + ow) * channel_count;

            float mean = 0.0f;
            for (uint32_t c = 0; c < channel_count; ++c)
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
