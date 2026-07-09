#pragma once

#include "netkit_config.h"

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED

#include <arm_nnfunctions.h>

#include <cstdint>

namespace CmsisQuantPlan
{

struct Conv2DCmsisHoist
{
    cmsis_nn_conv_params conv{};
    cmsis_nn_per_channel_quant_params quant{};
    cmsis_nn_dims input{};
    cmsis_nn_dims filter{};
    cmsis_nn_dims bias{};
    cmsis_nn_dims output{};
    bool ready = false;
};

struct DepthwiseConv2DCmsisHoist
{
    cmsis_nn_dw_conv_params dw_conv{};
    cmsis_nn_per_channel_quant_params quant{};
    cmsis_nn_dims input{};
    cmsis_nn_dims filter{};
    cmsis_nn_dims bias{};
    cmsis_nn_dims output{};
    bool ready = false;
};

struct Pool2DCmsisHoist
{
    cmsis_nn_pool_params pool{};
    cmsis_nn_dims input{};
    cmsis_nn_dims filter{};
    cmsis_nn_dims output{};
    bool ready = false;
};

struct FcCmsisHoist
{
    cmsis_nn_fc_params fc{};
    // Points at FcPlan::multipliers/shifts (per-channel or length-1).
    cmsis_nn_quant_params quant{};
    cmsis_nn_dims input{};
    cmsis_nn_dims filter{};
    cmsis_nn_dims bias{};
    cmsis_nn_dims output{};
    bool ready = false;
};

}  // namespace CmsisQuantPlan

#define NETKIT_CMSIS_PLAN_HOIST 1

#else

#define NETKIT_CMSIS_PLAN_HOIST 0

#endif
