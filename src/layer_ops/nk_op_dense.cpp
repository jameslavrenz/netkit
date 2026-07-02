#include "layer_ops/nk_dense_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanDense(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_features = block.dense.weights.shape[0];
    BumpMaxActivation(plan, out_features);
    plan.w = out_features;
    return true;
}

bool NkPrepareDense(const NkCnnOpContext& ctx)
{
    const uint32_t out_features = ctx.block.dense.weights.shape[0];
    ctx.output = View2D(ctx.write_buffer, 1, out_features);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalDense(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.dense.forward(input, output);
}
