#include "layer_ops/nk_depthwise_conv2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanDepthwiseConv2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const auto& dw = block.depthwise_conv.depthwise;
    const int pad_h_end = dw.pad_h_end >= 0 ? dw.pad_h_end : dw.pad_h;
    const int pad_w_end = dw.pad_w_end >= 0 ? dw.pad_w_end : dw.pad_w;
    const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
        plan.h, dw.kernel_h, dw.stride, dw.pad_h, pad_h_end);
    const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
        plan.w, dw.kernel_w, dw.stride, dw.pad_w, pad_w_end);
    const uint32_t out_c = static_cast<uint32_t>(dw.channels);
    BumpMaxActivation(plan, out_h * out_w * out_c);
    plan.h = out_h;
    plan.w = out_w;
    plan.channels = out_c;
    return true;
}

bool NkPrepareDepthwiseConv2D(const NkCnnOpContext& ctx)
{
    const auto& dw = ctx.block.depthwise_conv.depthwise;
    const int pad_h_end = dw.pad_h_end >= 0 ? dw.pad_h_end : dw.pad_h;
    const int pad_w_end = dw.pad_w_end >= 0 ? dw.pad_w_end : dw.pad_w;
    const uint32_t out_h = nk_op_detail::CalcOutputDimAsymmetric(
        ctx.input.shape[0], dw.kernel_h, dw.stride, dw.pad_h, pad_h_end);
    const uint32_t out_w = nk_op_detail::CalcOutputDimAsymmetric(
        ctx.input.shape[1], dw.kernel_w, dw.stride, dw.pad_w, pad_w_end);
    const uint32_t out_c = static_cast<uint32_t>(dw.channels);
    const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalDepthwiseConv2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.depthwise_conv.forward(input, output);
}
