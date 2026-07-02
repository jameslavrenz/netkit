#include "layer_ops/nk_flatten_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanFlatten(CnnBlock& /*block*/, NkCnnSpatialPlan& plan)
{
    const uint32_t features = plan.h * plan.w * plan.channels;
    BumpMaxActivation(plan, features);
    plan.h = 1;
    plan.w = features;
    plan.channels = 1;
    return true;
}

bool NkPrepareFlatten(const NkCnnOpContext& ctx)
{
    const uint32_t features = ctx.input.num_elements;
    ctx.output = View2D(ctx.write_buffer, 1, features);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalFlatten(CnnBlock& /*block*/, const Tensor& input, Tensor& output)
{
    FlattenNhwc(input, output);
}
