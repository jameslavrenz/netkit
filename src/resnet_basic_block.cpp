#include "resnet_basic_block.hpp"

#include "conv2d.hpp"
#include "fused_kernel_ops.hpp"
#include "nk_op_detail.hpp"
#include "tensor_access.hpp"

#include <cstring>

namespace
{
    using nk_op_detail::CalcOutputDim;
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

    Tensor work_a_tensor = fused_ops::NhwcView(work_a, out_h, out_w, out_c);
    conv1.forward(input, work_a_tensor);
    fused_ops::BatchNormInPlace(work_a_tensor, out_channels, bn1_scale, bn1_bias);
    fused_ops::ReluInPlace(work_a_tensor);

    Conv2D conv2{};
    conv2.kernel_size = 3;
    conv2.stride = 1;
    conv2.pad_h = 1;
    conv2.pad_w = 1;
    conv2.in_channels = out_channels;
    conv2.out_channels = out_channels;
    conv2.weights = conv2_weights;
    conv2.bias = conv2_bias;

    Tensor work_b_tensor = fused_ops::NhwcView(work_b, out_h, out_w, out_c);
    conv2.forward(work_a_tensor, work_b_tensor);
    fused_ops::BatchNormInPlace(work_b_tensor, out_channels, bn2_scale, bn2_bias);

    Tensor out_tensor = fused_ops::NhwcView(tensor_data_f32(output), out_h, out_w, out_c);

    if (has_identity_shortcut())
    {
        if (input.num_elements != out_elems)
            return;
        fused_ops::MatAddThenRelu(work_b_tensor, input, out_tensor);
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

        Tensor residual_tensor = fused_ops::NhwcView(residual, out_h, out_w, out_c);
        shortcut.forward(input, residual_tensor);
        fused_ops::BatchNormInPlace(residual_tensor, out_channels, shortcut_bn_scale, shortcut_bn_bias);

        fused_ops::MatAddThenRelu(work_b_tensor, residual_tensor, out_tensor);
    }
}
