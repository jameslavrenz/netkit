#include "layer_op_registry.hpp"

const NkOpsResolver& GetDefaultOpsResolver()
{
    return NkAllLayerOps::View();
}
