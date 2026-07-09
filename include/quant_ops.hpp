#pragma once

#include "layer_quant.hpp"
#include "nk_format.hpp"
#include "quant_output.hpp"
#include "tensor.hpp"

#include <cstdint>

namespace QuantOps
{
    int8_t QuantizeFloat(float value, float scale, int32_t zero_point);

    float DequantizeInt8(int8_t value, float scale, int32_t zero_point);

    void RequantizeToInt8(const float* values,
                          uint32_t count,
                          float output_scale,
                          int32_t output_zero_point,
                          int8_t* output);

    // Symmetric per-tensor int8 FC: input/weights may use non-zero zero-points.
    void FullyConnectedQuant(const int8_t* input,
                             uint32_t batch,
                             uint32_t in_features,
                             const int8_t* weights,
                             const int32_t* bias,
                             uint32_t out_features,
                             const NkFormat::MlpLayerQuantDesc& quant,
                             bool apply_relu,
                             int8_t* output_int8,
                             float* output_float);

    // Float input boundary -> int8 activations or float logits.
    void ForwardQuantizedDense(const Tensor& input,
                               const Tensor& weights,
                               const Tensor& bias,
                               const LayerQuant& quant,
                               bool apply_relu,
                               bool apply_softmax,
                               int8_t* quant_scratch,
                               bool input_is_float,
                               Tensor& output);

    // NHWC int8 conv; weights [out_c, kernel, kernel, in_c] from .nk catalog.
    void Conv2dNhwcQuant(const int8_t* input,
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

    // NHWC per-tensor int8 depthwise conv; weights [channels, kernel_h, kernel_w]
    // (output channel c reduces only over input channel c). Mirrors the float
    // DepthwiseConv2D layout and the int8 Conv2dNhwcQuant requantize path.
    void DepthwiseConv2dNhwcQuant(const int8_t* input,
                                  uint32_t in_h,
                                  uint32_t in_w,
                                  uint32_t channels,
                                  const int8_t* weights,
                                  const int32_t* bias,
                                  int kernel_h,
                                  int kernel_w,
                                  int stride,
                                  int pad_h,
                                  int pad_w,
                                  int pad_h_end,
                                  int pad_w_end,
                                  const NkFormat::MlpLayerQuantDesc& quant,
                                  bool apply_relu,
                                  int8_t* output);

    // Requantizing int8 elementwise add (per-tensor scales). Used by UIB residual.
    void ElementwiseAddS8(const int8_t* input1,
                          const int8_t* input2,
                          uint32_t count,
                          float input1_scale,
                          int32_t input1_zero_point,
                          float input2_scale,
                          int32_t input2_zero_point,
                          float output_scale,
                          int32_t output_zero_point,
                          int8_t* output);

    void AvgPool2dNhwcQuant(const int8_t* input,
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
                            float input_scale,
                            int32_t input_zero_point,
                            float output_scale,
                            int32_t output_zero_point,
                            int8_t* output);

    void MaxPool2dNhwcQuant(const int8_t* input,
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

    void FlattenNhwcInt8(const int8_t* input, uint32_t num_elements, int8_t* output);
}
