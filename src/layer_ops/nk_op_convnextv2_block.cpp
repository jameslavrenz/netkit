#include "layer_ops/nk_convnextv2_block_op.hpp"

#include "cmsis_buffer_size.hpp"
#include "cnn.hpp"
#include "convnextv2_block.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanConvNeXtV2Block(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t channels = static_cast<uint32_t>(block.convnextv2_block.block.channels);
    const uint32_t expanded = channels * static_cast<uint32_t>(ConvNeXtV2Block::kMlpRatio);
    const uint32_t spatial = plan.h * plan.w;
    CmsisBumpDepthwiseConv2dWorkspace(plan.h,
                                      plan.w,
                                      ConvNeXtV2Block::kDwKernel,
                                      ConvNeXtV2Block::kDwKernel,
                                      1,
                                      ConvNeXtV2Block::kDwPad,
                                      ConvNeXtV2Block::kDwPad,
                                      block.convnextv2_block.block.channels);
    BumpMaxActivation(plan, spatial * channels);
    BumpMaxActivation(plan, spatial * expanded);
    CmsisBumpGeluWorkspace(spatial * expanded);
    return true;
}

bool NkPrepareConvNeXtV2Block(const NkCnnOpContext& ctx)
{
    const std::array<uint32_t, 3> shape = {ctx.input.shape[0], ctx.input.shape[1],
                                           static_cast<uint32_t>(ctx.block.convnextv2_block.block.channels)};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalConvNeXtV2Block(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.convnextv2_block.forward(input, output);
}
