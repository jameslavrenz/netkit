#pragma once

#include <cstdint>

namespace CmsisDspUtil
{
// Vector copy (arm_copy_q7 / arm_copy_f32 when NETKIT_USE_CMSIS_DSP).
void CopyInt8(const int8_t* src, int8_t* dst, uint32_t count);
void CopyF32(const float* src, float* dst, uint32_t count);

// Index of maximum element.
uint32_t ArgMaxInt8(const int8_t* values, uint32_t count);
uint32_t ArgMaxF32(const float* values, uint32_t count);

// Contiguous dot product (arm_dot_prod_f32 when NETKIT_USE_CMSIS_DSP).
float DotProductF32(const float* a, const float* b, uint32_t count);

void MulF32(const float* a, const float* b, float* c, uint32_t count);
void AddF32(const float* a, const float* b, float* c, uint32_t count);
void MulScalarF32(const float* a, float scalar, float* c, uint32_t count);
void ScaleF32(float* c, float scalar, uint32_t count);
}  // namespace CmsisDspUtil

// Back-compat aliases for quant call sites.
namespace CmsisQuantUtil
{
using CmsisDspUtil::ArgMaxInt8;
using CmsisDspUtil::CopyInt8;
}  // namespace CmsisQuantUtil
