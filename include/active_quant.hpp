#pragma once

#include "netkit_config.h"

#if defined(NETKIT_USE_ESP_NN) && NETKIT_USE_ESP_NN && NETKIT_ESP_NN_ALLOWED
#include "esp_nn_quant.hpp"
namespace ActiveQuant = EspNnQuant;
#elif defined(NETKIT_USE_NMSIS_NN) && NETKIT_USE_NMSIS_NN && NETKIT_NMSIS_NN_ALLOWED
#include "nmsis_nn_quant.hpp"
namespace ActiveQuant = NmsisNnQuant;
#elif defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
#include "cmsis_nn_quant.hpp"
namespace ActiveQuant = CmsisNnQuant;
#else
/* No MCU NN accel — stubs still provide Try*/Finalize* (return false / no-op). */
#include "cmsis_nn_quant.hpp"
namespace ActiveQuant = CmsisNnQuant;
#endif
