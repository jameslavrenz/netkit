/*
 * trim_firmware.cpp — firmware recipe: link only the layer-op TUs you need.
 *
 * Build (conv + max-pool + flatten + dense — enough for cnn_4x4_single.nk):
 *   clang++ -std=c++26 -Iinclude -c examples/trim_firmware.cpp -o trim_firmware.o
 *   ar rcs libnetkit_trim.a ... src/layer_ops/nk_op_{conv2d,max_pool2d,flatten,dense}.cpp ...
 *   clang++ -std=c++26 -o trim_firmware trim_firmware.o libnetkit_trim.a
 *
 * Or: make trim-lib && see docs/BUILD_TARGETS.md
 */
#include "layer_ops/nk_conv2d_op.hpp"
#include "layer_ops/nk_dense_op.hpp"
#include "layer_ops/nk_flatten_op.hpp"
#include "layer_ops/nk_max_pool2d_op.hpp"
#include "ops_resolver.hpp"

#include <cstdio>

using TrimOps = NkOpList<NkConv2DOpDescriptor,
                         NkMaxPool2DOpDescriptor,
                         NkFlattenOpDescriptor,
                         NkDenseOpDescriptor>;

int main()
{
    const NkOpsResolver& resolver = TrimOps::View();
    std::printf("trim resolver: %u ops registered\n", resolver.num_registrations);
    return resolver.num_registrations == 4 ? 0 : 1;
}
