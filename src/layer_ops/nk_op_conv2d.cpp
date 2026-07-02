#include "layer_ops/nk_conv2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanConv2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h =
        CalcOutputDim(plan.h, block.conv.conv.kernel_size, block.conv.conv.stride, block.conv.conv.pad_h);
    const uint32_t out_w =
        CalcOutputDim(plan.w, block.conv.conv.kernel_size, block.conv.conv.stride, block.conv.conv.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(block.conv.conv.out_channels);
    BumpMaxActivation(plan, out_h * out_w * out_c);
    plan.h = out_h;
    plan.w = out_w;
    plan.channels = out_c;
    return true;
}

bool NkPrepareConv2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h = CalcOutputDim(ctx.input.shape[0],
                                         ctx.block.conv.conv.kernel_size,
                                         ctx.block.conv.conv.stride,
                                         ctx.block.conv.conv.pad_h);
    const uint32_t out_w = CalcOutputDim(ctx.input.shape[1],
                                         ctx.block.conv.conv.kernel_size,
                                         ctx.block.conv.conv.stride,
                                         ctx.block.conv.conv.pad_w);
    const uint32_t out_c = static_cast<uint32_t>(ctx.block.conv.conv.out_channels);
    const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalConv2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.conv.forward(input, output);
}
