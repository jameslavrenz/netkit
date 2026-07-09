#pragma once

#include "kernel_dispatch.hpp"
#include "netkit_config.h"
#include "reference_kernel.hpp"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include "cmsis_nn_kernel.hpp"
#endif

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
#include "xnnpack_kernel.hpp"
#endif

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
#include "cmsis_dsp_kernel.hpp"
#endif

// Backend composition:
//   MCU:  CMSIS-NN (LayerFast) + optional CMSIS-DSP (VectorFast)
//   CPU/MPU: XNNPACK (LayerFast) when enabled + optional CMSIS-DSP (VectorFast)
//   else: reference (with optional CMSIS-DSP as VectorFast)
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
    #if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
using Kernels = detail::ComposedKernel<CmsisDspKernel, CmsisNnKernel>;
    #else
using Kernels = detail::ComposedKernel<ReferenceKernel, CmsisNnKernel>;
    #endif
#elif defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
    #if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
using Kernels = detail::ComposedKernel<CmsisDspKernel, XnnpackKernel>;
    #else
using Kernels = detail::ComposedKernel<ReferenceKernel, XnnpackKernel>;
    #endif
#elif defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP
using Kernels = detail::ComposedKernel<CmsisDspKernel, ReferenceKernel>;
#else
using Kernels = ReferenceKernel;
#endif
