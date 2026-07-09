#pragma once

#include "netkit_config.h"
#include "nk_format.hpp"
#include "quant_plan_types.hpp"

#include <cstdint>

class Arena;

namespace CmsisNnQuant
{
#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN && NETKIT_CMSIS_NN_ALLOWED
constexpr bool kEnabled = true;
#else
constexpr bool kEnabled = false;
#endif

bool TryConv2dNhwcQuantPlan(const CmsisQuantPlan::Conv2DPlan& plan,
                            const int8_t* input,
                            const int8_t* weights,
                            const int32_t* bias,
                            int8_t* output);

bool TryDepthwiseConv2dNhwcQuantPlan(const CmsisQuantPlan::DepthwiseConv2DPlan& plan,
                                     const int8_t* input,
                                     const int8_t* weights,
                                     const int32_t* bias,
                                     int8_t* output);

bool TryMaxPool2dNhwcQuantPlan(const CmsisQuantPlan::Pool2DPlan& plan,
                               const int8_t* input,
                               int8_t* output);

bool TryFullyConnectedQuantPlan(const CmsisQuantPlan::FcPlan& plan,
                                const int8_t* input,
                                const int8_t* weights,
                                const int32_t* bias,
                                int8_t* output_int8);

bool TrySoftmaxS8Plan(const CmsisQuantPlan::SoftmaxPlan& plan, const int8_t* input, int8_t* output);

void FinalizeConv2DPlan(CmsisQuantPlan::Conv2DPlan& plan);
void FinalizeDepthwiseConv2DPlan(CmsisQuantPlan::DepthwiseConv2DPlan& plan);
void FinalizePool2DPlan(CmsisQuantPlan::Pool2DPlan& plan);
bool FinalizeFcPlan(CmsisQuantPlan::FcPlan& plan,
                    const int8_t* weights,
                    const int32_t* bias,
                    Arena& arena);

bool TryConv2dNhwcQuant(const int8_t* input,
                        uint32_t in_h,
                        uint32_t in_w,
                        uint32_t in_c,
                        const int8_t* weights,
                        const int32_t* bias,
                        int kernel_size,
                        int stride,
                        int pad_h,
                        int pad_w,
                        int pad_h_end,
                        int pad_w_end,
                        int out_channels,
                        const NkFormat::MlpLayerQuantDesc& quant,
                        bool apply_relu,
                        int8_t* output);

bool TryMaxPool2dNhwcQuant(const int8_t* input,
                           uint32_t in_h,
                           uint32_t in_w,
                           uint32_t in_c,
                           int pool_h,
                           int pool_w,
                           int stride,
                           int pad_h,
                           int pad_w,
                           int pad_h_end,
                           int pad_w_end,
                           int8_t* output);

bool TryFullyConnectedQuant(const int8_t* input,
                            uint32_t batch,
                            uint32_t in_features,
                            const int8_t* weights,
                            const int32_t* bias,
                            uint32_t out_features,
                            const NkFormat::MlpLayerQuantDesc& quant,
                            bool apply_relu,
                            int8_t* output_int8,
                            float* output_float);

// CMSIS int8 FC -> float logits (uses logits_i8 scratch; not for in-place with input_i8).
bool TryFullyConnectedQuantToFloat(const int8_t* input,
                                   uint32_t batch,
                                   uint32_t in_features,
                                   const int8_t* weights,
                                   const int32_t* bias,
                                   uint32_t out_features,
                                   const NkFormat::MlpLayerQuantDesc& quant,
                                   bool apply_relu,
                                   int8_t* logits_i8,
                                   float* output_float);

bool TrySoftmaxS8(const int8_t* input,
                  uint32_t num_rows,
                  uint32_t row_size,
                  float logit_scale,
                  int8_t* output);

bool TryElementwiseAddS8(const int8_t* input1,
                         const int8_t* input2,
                         uint32_t count,
                         float input1_scale,
                         int32_t input1_zero_point,
                         float input2_scale,
                         int32_t input2_zero_point,
                         float output_scale,
                         int32_t output_zero_point,
                         int8_t* output);

}  // namespace CmsisNnQuant
