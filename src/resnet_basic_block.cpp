#include "resnet_basic_block.hpp"

#include "conv2d.hpp"
#include "nk_op_detail.hpp"
#include "tensor_access.hpp"
#include "tensor_factory.hpp"

#include <algorithm>
#include <cstring>

namespace
{
    using nk_op_detail::CalcOutputDim;

    void BatchNormNhwcInPlace(float* data, uint32_t count, uint32_t channels, const float* scale, const float* bias)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t c = i % channels;
            data[i] = data[i] * scale[c] + bias[c];
        }
    }

    void ReluInPlace(float* data, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i)
            data[i] = std::max(0.0f, data[i]);
    }

    Tensor MakeView(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        const std::array<uint32_t, 3> shape = {h, w, c};
        return TensorFactory::ViewND(data, 3, shape);
    }
}

bool ResNetBasicBlock::has_identity_shortcut() const
{
    return stride == 1 && in_channels == out_channels;
}

void ResNetBasicBlock::output_spatial(uint32_t in_h, uint32_t in_w, uint32_t& out_h, uint32_t& out_w) const
{
    constexpr int kKernel = 3;
    constexpr int kPad = 1;
    out_h = CalcOutputDim(in_h, kKernel, stride, kPad);
    out_w = CalcOutputDim(in_w, kKernel, stride, kPad);
}

void ResNetBasicBlock::forward(const Tensor& input, Tensor& output)
{
    const uint32_t in_h = input.shape[0];
    const uint32_t in_w = input.shape[1];
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    output_spatial(in_h, in_w, out_h, out_w);

    const uint32_t out_c = static_cast<uint32_t>(out_channels);
    const uint32_t out_elems = out_h * out_w * out_c;
    const uint32_t scratch_needed =
        2u * out_elems + (has_identity_shortcut() ? 0u : out_elems);

    if (!scratch || scratch_elems < scratch_needed || !output.data)
        return;

    float* work_a = scratch;
    float* work_b = scratch + out_elems;
    float* residual = work_b + out_elems;

    Conv2D conv1{};
    conv1.kernel_size = 3;
    conv1.stride = stride;
    conv1.pad_h = 1;
    conv1.pad_w = 1;
    conv1.in_channels = in_channels;
    conv1.out_channels = out_channels;
    conv1.weights = conv1_weights;
    conv1.bias = conv1_bias;

    Tensor work_a_tensor = MakeView(work_a, out_h, out_w, out_c);
    conv1.forward(input, work_a_tensor);
    BatchNormNhwcInPlace(work_a, out_elems, out_c, bn1_scale, bn1_bias);
    ReluInPlace(work_a, out_elems);

    Conv2D conv2{};
    conv2.kernel_size = 3;
    conv2.stride = 1;
    conv2.pad_h = 1;
    conv2.pad_w = 1;
    conv2.in_channels = out_channels;
    conv2.out_channels = out_channels;
    conv2.weights = conv2_weights;
    conv2.bias = conv2_bias;

    Tensor work_b_tensor = MakeView(work_b, out_h, out_w, out_c);
    conv2.forward(work_a_tensor, work_b_tensor);
    BatchNormNhwcInPlace(work_b, out_elems, out_c, bn2_scale, bn2_bias);

    float* out = tensor_data_f32(output);

    if (has_identity_shortcut())
    {
        const float* inp = tensor_data_f32(const_cast<Tensor&>(input));
        if (input.num_elements != out_elems)
            return;
        for (uint32_t i = 0; i < out_elems; ++i)
            out[i] = work_b[i] + inp[i];
    }
    else
    {
        Conv2D shortcut{};
        shortcut.kernel_size = 1;
        shortcut.stride = stride;
        shortcut.pad_h = 0;
        shortcut.pad_w = 0;
        shortcut.in_channels = in_channels;
        shortcut.out_channels = out_channels;
        shortcut.weights = shortcut_weights;
        shortcut.bias = shortcut_bias;

        Tensor residual_tensor = MakeView(residual, out_h, out_w, out_c);
        shortcut.forward(input, residual_tensor);
        BatchNormNhwcInPlace(residual, out_elems, out_c, shortcut_bn_scale, shortcut_bn_bias);

        for (uint32_t i = 0; i < out_elems; ++i)
            out[i] = work_b[i] + residual[i];
    }

    ReluInPlace(out, out_elems);
}
