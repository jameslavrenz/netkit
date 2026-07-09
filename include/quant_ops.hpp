#pragma once

#include "layer_quant.hpp"
#include "nk_format.hpp"
#include "quant_integer.hpp"
#include "quant_output.hpp"
#include "tensor.hpp"

#include <cstdint>

namespace QuantOps
{
    // Symmetric per-tensor int8 FC: input/weights may use non-zero zero-points.
    // Float↔int8 conversion is Python-only (export / offline); C++ stays int8 end-to-end.
    void FullyConnectedQuant(const int8_t* input,
                             uint32_t batch,
                             uint32_t in_features,
                             const int8_t* weights,
                             const int32_t* bias,
                             uint32_t out_features,
                             const NkFormat::MlpLayerQuantDesc& quant,
                             QuantInteger::QuantClamp clamp,
                             int8_t* output_int8);

    // Convenience: ReLU clamp or none.
    inline void FullyConnectedQuant(const int8_t* input,
                                    uint32_t batch,
                                    uint32_t in_features,
                                    const int8_t* weights,
                                    const int32_t* bias,
                                    uint32_t out_features,
                                    const NkFormat::MlpLayerQuantDesc& quant,
                                    bool apply_relu,
                                    int8_t* output_int8)
    {
        FullyConnectedQuant(input,
                            batch,
                            in_features,
                            weights,
                            bias,
                            out_features,
                            quant,
                            apply_relu ? QuantInteger::QuantClamp::ReLU
                                       : QuantInteger::QuantClamp::None,
                            output_int8);
    }

    // Int8 dense + optional ReLU/softmax. Input tensor must be Int8.
    void ForwardQuantizedDense(const Tensor& input,
                               const Tensor& weights,
                               const Tensor& bias,
                               const LayerQuant& quant,
                               bool apply_relu,
                               bool apply_softmax,
                               int8_t* quant_scratch,
                               Tensor& output);

    // Optional residual for int8 conv epilogue (UIB projection). Scales are
    // residual tensor's; output scale/zp match the conv quant desc.
    struct ResidualAddS8
    {
        const int8_t* data = nullptr;
        float scale = 1.0f;
        int32_t zero_point = 0;
    };

    // NHWC int8 conv; weights [out_c, kernel, kernel, in_c] from .nk catalog.
    // residual: if non-null, ElementwiseAddS8(output, residual) after conv
    // (CMSIS-NN and reference both use the same epilogue — no native conv+add).
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
                         QuantInteger::QuantClamp clamp,
                         int8_t* output,
                         const ResidualAddS8* residual = nullptr);

    inline void Conv2dNhwcQuant(const int8_t* input,
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
                                int8_t* output,
                                const ResidualAddS8* residual = nullptr)
    {
        Conv2dNhwcQuant(input,
                        in_h,
                        in_w,
                        in_c,
                        weights,
                        bias,
                        kernel_size,
                        stride,
                        pad_h,
                        pad_w,
                        pad_h_end,
                        pad_w_end,
                        out_channels,
                        quant,
                        apply_relu ? QuantInteger::QuantClamp::ReLU : QuantInteger::QuantClamp::None,
                        output,
                        residual);
    }

    // NHWC per-tensor int8 depthwise conv; weights [channels, kernel_h, kernel_w]
    // (output channel c reduces only over input channel c). Mirrors the float
    // DepthwiseConv2D layout. Int8→int8 uses integer multiply-by-quantized-multiplier.
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
                                  QuantInteger::QuantClamp clamp,
                                  int8_t* output);

    inline void DepthwiseConv2dNhwcQuant(const int8_t* input,
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
                                         int8_t* output)
    {
        DepthwiseConv2dNhwcQuant(input,
                                 in_h,
                                 in_w,
                                 channels,
                                 weights,
                                 bias,
                                 kernel_h,
                                 kernel_w,
                                 stride,
                                 pad_h,
                                 pad_w,
                                 pad_h_end,
                                 pad_w_end,
                                 quant,
                                 apply_relu ? QuantInteger::QuantClamp::ReLU
                                            : QuantInteger::QuantClamp::None,
                                 output);
    }

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
