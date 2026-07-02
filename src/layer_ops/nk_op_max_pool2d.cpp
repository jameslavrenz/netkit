#include "layer_ops/nk_max_pool2d_op.hpp"

#include "cnn.hpp"
#include "nk_op_detail.hpp"
#include "tensor_factory.hpp"
#include <array>

using namespace TensorFactory;
using namespace nk_op_detail;

bool NkPlanMaxPool2D(CnnBlock& block, NkCnnSpatialPlan& plan)
{
    const uint32_t out_h = CalcOutputDim(plan.h, block.pool.pool_size, block.pool.stride, block.pool.pad_h);
    const uint32_t out_w = CalcOutputDim(plan.w, block.pool.pool_size, block.pool.stride, block.pool.pad_w);
    BumpMaxActivation(plan, out_h * out_w * plan.channels);
    plan.h = out_h;
    plan.w = out_w;
    return true;
}

bool NkPrepareMaxPool2D(const NkCnnOpContext& ctx)
{
    const uint32_t out_h =
        CalcOutputDim(ctx.input.shape[0], ctx.block.pool.pool_size, ctx.block.pool.stride, ctx.block.pool.pad_h);
    const uint32_t out_w =
        CalcOutputDim(ctx.input.shape[1], ctx.block.pool.pool_size, ctx.block.pool.stride, ctx.block.pool.pad_w);
    const std::array<uint32_t, 3> shape = {out_h, out_w, ctx.input.shape[2]};
    ctx.output = ViewND(ctx.write_buffer, 3, shape);
    return ctx.output.data != nullptr && ctx.output.num_elements <= ctx.max_activation_elements;
}

void NkEvalMaxPool2D(CnnBlock& block, const Tensor& input, Tensor& output)
{
    block.pool.forward(input, output);
}
