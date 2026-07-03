#include "reference_kernel.hpp"
#include "tensor_access.hpp"
#include <cmath>
#include <cfloat>

void ReferenceKernel::MulImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = a_data[i] * b_data[i];
}

void ReferenceKernel::MulScalarImpl(const Tensor& a, float scalar, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = a_data[i] * scalar;
}

void ReferenceKernel::MatAddImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    const uint32_t rows = a.shape[0];
    const uint32_t cols = a.shape[1];

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
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t idx = 0; idx < a.num_elements; idx++)
        c_data[idx] = a_data[idx] + b_data[idx];
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
        for (uint32_t j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (uint32_t t = 0; t < k; t++)
            {
                const uint32_t a_index = i * a.stride[0] + t * a.stride[1];
                const uint32_t b_index = t * b.stride[0] + j * b.stride[1];
                sum += a_data[a_index] * b_data[b_index];
            }
            const uint32_t c_index = i * c.stride[0] + j * c.stride[1];
            c_data[c_index] = sum;
        }
    }
}

void ReferenceKernel::MulNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    const float* b_data = static_cast<const float*>(b.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t idx = 0; idx < a.num_elements; idx++)
        c_data[idx] = a_data[idx] * b_data[idx];
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
        for (uint32_t oc = 0; oc < out_features; ++oc)
        {
            float sum = 0.0f;
            for (uint32_t ic = 0; ic < in_features; ++ic)
            {
                const uint32_t in_index = b * input.stride[0] + ic * input.stride[1];
                const uint32_t wt_index = oc * kernel.stride[0] + ic * kernel.stride[1];
                sum += in[in_index] * wt[wt_index];
            }
            const uint32_t out_index = b * output.stride[0] + oc * output.stride[1];
            out[out_index] = sum;
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

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = (a_data[i] > 0.0f) ? a_data[i] : 0.0f;
}

void ReferenceKernel::SigmoidImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = 1.0f / (1.0f + std::expf(-a_data[i]));
}

void ReferenceKernel::TanhImpl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = std::tanhf(a_data[i]);
}

void ReferenceKernel::LeakyReLUImpl(const Tensor& a, Tensor& c, float alpha)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
        c_data[i] = (a_data[i] > 0.0f) ? a_data[i] : alpha * a_data[i];
}

void ReferenceKernel::ReLU6Impl(const Tensor& a, Tensor& c)
{
    const float* a_data = static_cast<const float*>(a.data);
    float* c_data = static_cast<float*>(c.data);

    for (uint32_t i = 0; i < a.num_elements; i++)
    {
        const float x = a_data[i];
        if (x < 0.0f)
            c_data[i] = 0.0f;
        else if (x > 6.0f)
            c_data[i] = 6.0f;
        else
            c_data[i] = x;
    }
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
    for (uint32_t i = 0; i < n; i++)
    {
        const float e = std::expf(a_data[i] - max_val);
        c_data[i] = e;
        sum += e;
    }

    const float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < n; i++)
        c_data[i] *= inv_sum;
}

bool ReferenceKernel::Conv2dForwardImpl(const Tensor& input,
                                        float* weights,
                                        float* bias,
                                        int kernel_size,
                                        int stride,
                                        int pad_h,
                                        int pad_w,
                                        int in_channels,
                                        int out_channels,
                                        NetkitKernelActivation /*fuse_activation*/,
                                        Tensor& output)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (int oc = 0; oc < out_channels; oc++)
    {
        for (uint32_t oh = 0; oh < out_h; oh++)
        {
            for (uint32_t ow = 0; ow < out_w; ow++)
            {
                float sum = bias ? bias[oc] : 0.0f;

                for (int kh = 0; kh < kernel_size; kh++)
                {
                    for (int kw = 0; kw < kernel_size; kw++)
                    {
                        for (int ic = 0; ic < in_channels; ic++)
                        {
                            const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                            const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                            if (ih < 0 || iw < 0 || ih >= static_cast<int>(input.shape[0]) ||
                                iw >= static_cast<int>(input.shape[1]))
                                continue;

                            const uint32_t in_idx =
                                index_nhwc(input, static_cast<uint32_t>(ih), static_cast<uint32_t>(iw), ic);
                            const uint32_t w_idx =
                                (((oc * kernel_size + kh) * kernel_size + kw) * in_channels) + ic;
                            sum += in[in_idx] * weights[w_idx];
                        }
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * static_cast<uint32_t>(out_channels) +
                                         static_cast<uint32_t>(oc);
                out[out_idx] = sum;
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
                                                 int channels,
                                                 NetkitKernelActivation /*fuse_activation*/,
                                                 Tensor& output)
{
    float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (int c = 0; c < channels; ++c)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                float sum = bias ? bias[c] : 0.0f;

                for (int kh = 0; kh < kernel_h; ++kh)
                {
                    for (int kw = 0; kw < kernel_w; ++kw)
                    {
                        const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (ih < 0 || iw < 0 || ih >= static_cast<int>(input.shape[0]) ||
                            iw >= static_cast<int>(input.shape[1]))
                            continue;

                        const uint32_t in_idx =
                            index_nhwc(input, static_cast<uint32_t>(ih), static_cast<uint32_t>(iw), c);
                        const uint32_t w_idx =
                            (static_cast<uint32_t>(c) * static_cast<uint32_t>(kernel_h) +
                             static_cast<uint32_t>(kh)) *
                                static_cast<uint32_t>(kernel_w) +
                            static_cast<uint32_t>(kw);
                        sum += in[in_idx] * weights[w_idx];
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * static_cast<uint32_t>(channels) +
                                         static_cast<uint32_t>(c);
                out[out_idx] = sum;
            }
        }
    }

    return false;
}

void ReferenceKernel::MaxPool2dForwardImpl(const Tensor& input,
                                          int pool_size,
                                          int stride,
                                          int pad_h,
                                          int pad_w,
                                          Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (uint32_t c = 0; c < channels; ++c)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                float max_val = -FLT_MAX;
                for (int kh = 0; kh < pool_size; ++kh)
                {
                    for (int kw = 0; kw < pool_size; ++kw)
                    {
                        const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (ih < 0 || iw < 0 || static_cast<uint32_t>(ih) >= in_h ||
                            static_cast<uint32_t>(iw) >= in_w)
                            continue;

                        const uint32_t in_idx = index_nhwc(input, static_cast<uint32_t>(ih),
                                                           static_cast<uint32_t>(iw), c);
                        if (in[in_idx] > max_val)
                            max_val = in[in_idx];
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * channels + c;
                out[out_idx] = max_val;
            }
        }
    }
}

void ReferenceKernel::AvgPool2dForwardImpl(const Tensor& input,
                                           int pool_size,
                                           int stride,
                                           int pad_h,
                                           int pad_w,
                                           Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (uint32_t c = 0; c < channels; ++c)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                float sum = 0.0f;
                uint32_t count = 0;
                for (int kh = 0; kh < pool_size; ++kh)
                {
                    for (int kw = 0; kw < pool_size; ++kw)
                    {
                        const int ih = static_cast<int>(oh) * stride + kh - pad_h;
                        const int iw = static_cast<int>(ow) * stride + kw - pad_w;
                        if (ih < 0 || iw < 0 || static_cast<uint32_t>(ih) >= in_h ||
                            static_cast<uint32_t>(iw) >= in_w)
                            continue;

                        const uint32_t in_idx = index_nhwc(input, static_cast<uint32_t>(ih),
                                                           static_cast<uint32_t>(iw), c);
                        sum += in[in_idx];
                        ++count;
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * channels + c;
                out[out_idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
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

    for (uint32_t i = 0; i < input.num_elements; ++i)
    {
        const uint32_t c = i % channels_u;
        out[i] = in[i] * scale[c] + bias[c];
    }
}
