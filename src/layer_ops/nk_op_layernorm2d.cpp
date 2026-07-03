#include "layer_ops/nk_layernorm2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanLayerNorm2d(CnnBlock& /*block*/, NkCnnSpatialPlan& plan)
{
    BumpMaxActivation(plan, plan.h * plan.w * plan.channels);
    return true;
}

bool NkPrepareLayerNorm2d(const NkCnnOpContext& ctx)
{
    const std::array<uint32_t, 3> shape = {ctx.input.shape[0], ctx.input.shape[1], ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalLayerNorm2d(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.layer_norm.forward(input, output);
}
