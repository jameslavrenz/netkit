/*
 * NMSIS-NN float LayerFast adapters.
 *
 * NMSIS-NN provides int8 kernels only. This TU is linked when NETKIT_NMSIS_NN is
 * effective so NmsisNnKernel symbols exist; every Try* returns false and
 * ComposedKernel falls back to ReferenceKernel (same pattern as an accel miss).
 */
#include "nmsis_nn_kernel.hpp"
#include "netkit_config.h"

#if defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && NETKIT_NMSIS_NN_ALLOWED

bool NmsisNnKernel::TryConv2dForward(const Tensor&,
                                   float*,
                                   float*,
                                   int,
                                   int,
                                   int,
                                   int,
                                   int,
                                   int,
                                   NetkitKernelActivation,
                                   Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryDepthwiseConv2dForward(const Tensor&,
                                            float*,
                                            float*,
                                            int,
                                            int,
                                            int,
                                            int,
                                            int,
                                            int,
                                            NetkitKernelActivation,
                                            Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryMaxPool2dForward(const Tensor&,
                                      int,
                                      int,
                                      int,
                                      int,
                                      NetkitKernelActivation,
                                      Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryAvgPool2dForward(const Tensor&, int, int, int, int, Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryBatchNorm2dForward(const Tensor&, const float*, const float*, int, Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryFullyConnectedWithBias(const Tensor&,
                                            const Tensor&,
                                            const Tensor&,
                                            NetkitKernelActivation,
                                            Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryActivationForward(const Tensor&,
                                       Tensor&,
                                       NetkitKernelActivation,
                                       float)
{
    return false;
}

bool NmsisNnKernel::TrySoftmaxForward(const Tensor&, Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryGeluForward(const Tensor&, Tensor&)
{
    return false;
}

bool NmsisNnKernel::TryMatAdd(const Tensor&, const Tensor&, Tensor&)
{
    return false;
}

#endif /* NETKIT_USE_NMSIS_NN */
