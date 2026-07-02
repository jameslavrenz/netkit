#pragma once

/*
 * Full layer-op registry — includes every descriptor.
 *
 * For trimmed firmware, include only the headers under layer_ops/ that you need
 * and build NkOpList<YourOps...>::View() with the matching src/layer_ops/*.cpp units.
 */

#include "layer_ops/nk_avg_pool2d_op.hpp"
#include "layer_ops/nk_batch_norm2d_op.hpp"
#include "layer_ops/nk_conv2d_op.hpp"
#include "layer_ops/nk_dense_op.hpp"
#include "layer_ops/nk_flatten_op.hpp"
#include "layer_ops/nk_max_pool2d_op.hpp"

using NkAllLayerOps = NkOpList<NkDenseOpDescriptor,
                                 NkConv2DOpDescriptor,
                                 NkMaxPool2DOpDescriptor,
                                 NkFlattenOpDescriptor,
                                 NkAvgPool2DOpDescriptor,
                                 NkBatchNorm2dOpDescriptor>;
