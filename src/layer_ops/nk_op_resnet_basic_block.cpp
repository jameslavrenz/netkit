#include "layer_ops/nk_resnet_basic_block_op.hpp"

#include "cnn.hpp"
#include "resnet_basic_block.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanResNetBasicBlock(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const ResNetBasicBlock& resblock = block.resnet_basic_block.block;
    uint32_t out_h = plan.h;
    uint32_t out_w = plan.w;
    resblock.output_spatial(plan.h, plan.w, out_h, out_w);

    BumpMaxActivation(plan, plan.h * plan.w * static_cast<uint32_t>(resblock.in_channels));
    BumpMaxActivation(plan, out_h * out_w * static_cast<uint32_t>(resblock.out_channels));

    plan.h = out_h;
    plan.w = out_w;
    plan.channels = static_cast<uint32_t>(resblock.out_channels);
    return true;
}

bool NkPrepareResNetBasicBlock(const NkCnnOpContext& ctx)
{
    const ResNetBasicBlock& resblock = ctx.block.resnet_basic_block.block;
    uint32_t out_h = ctx.input.shape[0];
    uint32_t out_w = ctx.input.shape[1];
    resblock.output_spatial(ctx.input.shape[0], ctx.input.shape[1], out_h, out_w);

    const std::array<uint32_t, 3> shape = {out_h, out_w, static_cast<uint32_t>(resblock.out_channels)};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalResNetBasicBlock(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.resnet_basic_block.forward(input, output);
}
