#pragma once

#include "kernel_dispatch.hpp"
#include "netkit_config.h"
#include "reference_kernel.hpp"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include "cmsis_nn_kernel.hpp"
#endif

#if defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && NETKIT_ESP_NN_ALLOWED
#include "esp_nn_kernel.hpp"
#endif

#if defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && NETKIT_NMSIS_NN_ALLOWED
#include "nmsis_nn_kernel.hpp"
#endif

#if defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
#include "xnnpack_kernel.hpp"
#endif

// Backend composition:
//   MCU (mcu_arm): CMSIS-NN LayerFast over reference VectorFast (int8 production path)
//   MCU (mcu_esp): ESP-NN LayerFast slot (float Try* always miss → reference; int8 via EspNnQuant)
//   MCU (mcu_risc): NMSIS-NN LayerFast slot (float Try* always miss → reference; int8 via NmsisNnQuant)
//   CPU/MPU:       XNNPACK LayerFast when enabled, else reference
//   No CMSIS-DSP — float32 MCU is reference-only except CMSIS float LayerFast on Arm
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, CmsisNnKernel>;
#elif defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && NETKIT_ESP_NN_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, EspNnKernel>;
#elif defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && NETKIT_NMSIS_NN_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, NmsisNnKernel>;
#elif defined(NETKIT_USE_XNNPACK) && NETKIT_USE_XNNPACK && NETKIT_XNNPACK_ALLOWED
using Kernels = detail::ComposedKernel<ReferenceKernel, XnnpackKernel>;
#else
using Kernels = ReferenceKernel;
#endif
