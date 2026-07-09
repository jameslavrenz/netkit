#pragma once

#include "kernel_crtp.hpp"
#include "kernel_activation.hpp"
#include "reference_kernel.hpp"
#include "tensor.hpp"
#include <cfloat>
#include <type_traits>

namespace detail
{
    template<typename T>
    concept NkAcceleratedKernel = !std::same_as<T, ReferenceKernel>;

    constexpr void ApplyReferenceActivation(const Tensor& a,
                                            Tensor& c,
                                            NetkitKernelActivation activation,
                                            float alpha)
    {
        switch (activation)
        {
            case NetkitKernelActivation::ReLU:
                ReferenceKernel::ReLUImpl(a, c);
                break;
            case NetkitKernelActivation::Sigmoid:
                ReferenceKernel::SigmoidImpl(a, c);
                break;
            case NetkitKernelActivation::Tanh:
                ReferenceKernel::TanhImpl(a, c);
                break;
            case NetkitKernelActivation::LeakyReLU:
                ReferenceKernel::LeakyReLUImpl(a, c, alpha);
                break;
            case NetkitKernelActivation::ReLU6:
                ReferenceKernel::ReLU6Impl(a, c);
                break;
            default:
                break;
        }
    }

    template<NkAcceleratedKernel Fast, typename Reference = ReferenceKernel>
    void TryVectorMul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if (!Fast::TryMul(a, b, c))
            Reference::MulImpl(a, b, c);
    }

    template<typename Reference = ReferenceKernel>
    void TryVectorMul(const Tensor& a, const Tensor& b, Tensor& c)
        requires std::same_as<Reference, ReferenceKernel>
    {
        Reference::MulImpl(a, b, c);
    }

    template<NkAcceleratedKernel Fast, typename Reference = ReferenceKernel>
    void TryVectorMulScalar(const Tensor& a, float scalar, Tensor& c)
    {
        if (!Fast::TryMulScalar(a, scalar, c))
            Reference::MulScalarImpl(a, scalar, c);
    }

    template<typename Reference = ReferenceKernel>
    void TryVectorMulScalar(const Tensor& a, float scalar, Tensor& c)
        requires std::same_as<Reference, ReferenceKernel>
    {
        Reference::MulScalarImpl(a, scalar, c);
    }

    template<NkAcceleratedKernel Fast, typename Reference = ReferenceKernel>
    void TryVectorMatMul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if (!Fast::TryMatMul(a, b, c))
            Reference::MatMulImpl(a, b, c);
    }

    template<typename Reference = ReferenceKernel>
    void TryVectorMatMul(const Tensor& a, const Tensor& b, Tensor& c)
        requires std::same_as<Reference, ReferenceKernel>
    {
        Reference::MatMulImpl(a, b, c);
    }

    template<typename LayerFast, typename VectorFast>
    void TryMatAdd2D(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (!LayerFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddImpl(a, b, c);
        }
        else if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (!VectorFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddImpl(a, b, c);
        }
        else
            ReferenceKernel::MatAddImpl(a, b, c);
    }

    template<typename LayerFast, typename VectorFast>
    void TryMatAddND(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (!LayerFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddNDImpl(a, b, c);
        }
        else if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (!VectorFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddNDImpl(a, b, c);
        }
        else
            ReferenceKernel::MatAddNDImpl(a, b, c);
    }

    template<typename LayerFast, typename VectorFast>
    void TryNnActivation(const Tensor& a,
                         Tensor& c,
                         NetkitKernelActivation activation,
                         float alpha)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (!LayerFast::TryActivationForward(a, c, activation, alpha))
                ApplyReferenceActivation(a, c, activation, alpha);
        }
        else if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (activation == NetkitKernelActivation::ReLU)
            {
                if (!VectorFast::TryClip(a, c, 0.0f, FLT_MAX))
                    ReferenceKernel::ReLUImpl(a, c);
            }
            else if (activation == NetkitKernelActivation::ReLU6)
            {
                if (!VectorFast::TryClip(a, c, 0.0f, 6.0f))
                    ReferenceKernel::ReLU6Impl(a, c);
            }
            else
                ApplyReferenceActivation(a, c, activation, alpha);
        }
        else
            ApplyReferenceActivation(a, c, activation, alpha);
    }

    template<typename LayerFast>
    bool TryLayerConv(const Tensor& input,
                      float* weights,
                      float* bias,
                      int kernel_size,
                      int stride,
                      int pad_h,
                      int pad_w,
                      int in_channels,
                      int out_channels,
                      NetkitKernelActivation fuse_activation,
                      Tensor& output)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (LayerFast::TryConv2dForward(input,
                                            weights,
                                            bias,
                                            kernel_size,
                                            stride,
                                            pad_h,
                                            pad_w,
                                            in_channels,
                                            out_channels,
                                            fuse_activation,
                                            output))
                return true;
        }
        return ReferenceKernel::Conv2dForwardImpl(input,
                                            weights,
                                            bias,
                                            kernel_size,
                                            stride,
                                            pad_h,
                                            pad_w,
                                            pad_h,
                                            pad_w,
                                            in_channels,
                                            out_channels,
                                            fuse_activation,
                                            output,
                                            nullptr);
    }

    template<typename LayerFast>
    bool TryLayerDepthwiseConv(const Tensor& input,
                               float* weights,
                               float* bias,
                               int kernel_h,
                               int kernel_w,
                               int stride,
                               int pad_h,
                               int pad_w,
                               int pad_h_end,
                               int pad_w_end,
                               int channels,
                               NetkitKernelActivation fuse_activation,
                               Tensor& output)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (pad_h_end == pad_h && pad_w_end == pad_w)
            {
                if (LayerFast::TryDepthwiseConv2dForward(input,
                                                         weights,
                                                         bias,
                                                         kernel_h,
                                                         kernel_w,
                                                         stride,
                                                         pad_h,
                                                         pad_w,
                                                         channels,
                                                         fuse_activation,
                                                         output))
                    return true;
            }
        }
        return ReferenceKernel::DepthwiseConv2dForwardImpl(input,
                                                           weights,
                                                           bias,
                                                           kernel_h,
                                                           kernel_w,
                                                           stride,
                                                           pad_h,
                                                           pad_w,
                                                           pad_h_end,
                                                           pad_w_end,
                                                           channels,
                                                           fuse_activation,
                                                           output);
    }

    template<typename LayerFast, typename VectorFast>
    bool TryFullyConnected(const Tensor& input,
                           const Tensor& weights,
                           const Tensor& bias,
                           NetkitKernelActivation fuse_activation,
                           Tensor& output)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (LayerFast::TryFullyConnectedWithBias(input, weights, bias, fuse_activation, output))
                return true;
        }
        else if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (VectorFast::TryFullyConnectedWithBias(input, weights, bias, output))
            {
                if (kernel_activation_is_fused(fuse_activation))
                {
                    if (fuse_activation == NetkitKernelActivation::ReLU)
                    {
                        if (VectorFast::TryClip(output, output, 0.0f, FLT_MAX))
                            return true;
                    }
                    else if (fuse_activation == NetkitKernelActivation::ReLU6)
                    {
                        if (VectorFast::TryClip(output, output, 0.0f, 6.0f))
                            return true;
                    }
                }
                return false;
            }
        }
        return ReferenceKernel::FullyConnectedWithBiasImpl(input, weights, bias, fuse_activation, output);
    }

    template<typename LayerFast, typename VectorFast>
    void TryGelu(const Tensor& a, Tensor& c)
    {
        if constexpr (NkAcceleratedKernel<LayerFast>)
        {
            if (LayerFast::TryGeluForward(a, c))
                return;
        }
        if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (VectorFast::TryGeluForward(a, c))
                return;
        }
        ReferenceKernel::GeluImpl(a, c);
    }

    template<typename VectorFast>
    void TryGrn2d(const Tensor& input,
                  const float* gamma,
                  const float* beta,
                  int channels,
                  float eps,
                  float* channel_norm_scratch,
                  Tensor& output)
    {
        if constexpr (NkAcceleratedKernel<VectorFast>)
        {
            if (VectorFast::TryGrn2dForward(
                    input, gamma, beta, channels, eps, channel_norm_scratch, output))
                return;
        }
        ReferenceKernel::Grn2dForwardImpl(
            input, gamma, beta, channels, eps, channel_norm_scratch, output);
    }

    /*
     * ComposedKernel<VectorFast, LayerFast> — single CRTP implementation for all backend mixes.
     * ReferenceKernel as a template argument means "use reference for that role".
     */
    template<typename VectorFast, typename LayerFast>
    struct ComposedKernel : KernelBase<ComposedKernel<VectorFast, LayerFast>>
    {
        static void MulImpl(const Tensor& a, const Tensor& b, Tensor& c)
        {
            TryVectorMul<VectorFast>(a, b, c);
        }

        static void MulScalarImpl(const Tensor& a, float scalar, Tensor& c)
        {
            TryVectorMulScalar<VectorFast>(a, scalar, c);
        }

        static void MatAddImpl(const Tensor& a, const Tensor& b, Tensor& c)
        {
            TryMatAdd2D<LayerFast, VectorFast>(a, b, c);
        }

        static void MatAddNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
        {
            TryMatAddND<LayerFast, VectorFast>(a, b, c);
        }

        static void MatMulImpl(const Tensor& a, const Tensor& b, Tensor& c)
        {
            TryVectorMatMul<VectorFast>(a, b, c);
        }

        static void MulNDImpl(const Tensor& a, const Tensor& b, Tensor& c)
        {
            TryVectorMul<VectorFast>(a, b, c);
        }

        static void ReLUImpl(const Tensor& a, Tensor& c)
        {
            TryNnActivation<LayerFast, VectorFast>(a, c, NetkitKernelActivation::ReLU, 0.0f);
        }

        static void SigmoidImpl(const Tensor& a, Tensor& c)
        {
            TryNnActivation<LayerFast, VectorFast>(a, c, NetkitKernelActivation::Sigmoid, 0.0f);
        }

        static void TanhImpl(const Tensor& a, Tensor& c)
        {
            TryNnActivation<LayerFast, VectorFast>(a, c, NetkitKernelActivation::Tanh, 0.0f);
        }

        static void LeakyReLUImpl(const Tensor& a, Tensor& c, float alpha)
        {
            TryNnActivation<LayerFast, VectorFast>(a, c, NetkitKernelActivation::LeakyReLU, alpha);
        }

        static void ReLU6Impl(const Tensor& a, Tensor& c)
        {
            TryNnActivation<LayerFast, VectorFast>(a, c, NetkitKernelActivation::ReLU6, 0.0f);
        }

        static void SoftmaxImpl(const Tensor& a, Tensor& c)
        {
            if constexpr (NkAcceleratedKernel<LayerFast>)
            {
                if (!LayerFast::TrySoftmaxForward(a, c))
                    ReferenceKernel::SoftmaxImpl(a, c);
            }
            else
                ReferenceKernel::SoftmaxImpl(a, c);
        }

        static void GeluImpl(const Tensor& a, Tensor& c)
        {
            TryGelu<LayerFast, VectorFast>(a, c);
        }

        static void Grn2dForwardImpl(const Tensor& input,
                                     const float* gamma,
                                     const float* beta,
                                     int channels,
                                     float eps,
                                     float* channel_norm_scratch,
                                     Tensor& output)
        {
            TryGrn2d<VectorFast>(input, gamma, beta, channels, eps, channel_norm_scratch, output);
        }

        static bool Conv2dForwardImpl(const Tensor& input,
                                      float* weights,
                                      float* bias,
                                      int kernel_size,
                                      int stride,
                                      int pad_h,
                                      int pad_w,
                                      int pad_h_end,
                                      int pad_w_end,
                                      int in_channels,
                                      int out_channels,
                                      NetkitKernelActivation fuse_activation,
                                      Tensor& output)
        {
            if constexpr (NkAcceleratedKernel<LayerFast>)
            {
                if (pad_h_end == pad_h && pad_w_end == pad_w)
                {
                    if (LayerFast::TryConv2dForward(input,
                                                    weights,
                                                    bias,
                                                    kernel_size,
                                                    stride,
                                                    pad_h,
                                                    pad_w,
                                                    in_channels,
                                                    out_channels,
                                                    fuse_activation,
                                                    output))
                        return true;
                }
            }
            return ReferenceKernel::Conv2dForwardImpl(input,
                                                      weights,
                                                      bias,
                                                      kernel_size,
                                                      stride,
                                                      pad_h,
                                                      pad_w,
                                                      pad_h_end,
                                                      pad_w_end,
                                                      in_channels,
                                                      out_channels,
                                                      fuse_activation,
                                                      output,
                                                      nullptr);
        }

        static bool DepthwiseConv2dForwardImpl(const Tensor& input,
                                               float* weights,
                                               float* bias,
                                               int kernel_h,
                                               int kernel_w,
                                               int stride,
                                               int pad_h,
                                               int pad_w,
                                               int pad_h_end,
                                               int pad_w_end,
                                               int channels,
                                               NetkitKernelActivation fuse_activation,
                                               Tensor& output)
        {
            return TryLayerDepthwiseConv<LayerFast>(input,
                                                    weights,
                                                    bias,
                                                    kernel_h,
                                                    kernel_w,
                                                    stride,
                                                    pad_h,
                                                    pad_w,
                                                    pad_h_end,
                                                    pad_w_end,
                                                    channels,
                                                    fuse_activation,
                                                    output);
        }

        static bool MaxPool2dForwardImpl(const Tensor& input,
                                         int pool_h,
                                         int pool_w,
                                         int stride,
                                         int pad_h,
                                         int pad_w,
                                         int pad_h_end,
                                         int pad_w_end,
                                         NetkitKernelActivation fuse_activation,
                                         Tensor& output)
        {
            if constexpr (NkAcceleratedKernel<LayerFast>)
            {
                if (pad_h_end == pad_h && pad_w_end == pad_w && pool_h == pool_w)
                {
                    if (LayerFast::TryMaxPool2dForward(input,
                                                       pool_h,
                                                       stride,
                                                       pad_h,
                                                       pad_w,
                                                       fuse_activation,
                                                       output))
                        return kernel_activation_is_fused(fuse_activation);
                }
            }
            ReferenceKernel::MaxPool2dForwardImpl(input,
                                                  pool_h,
                                                  pool_w,
                                                  stride,
                                                  pad_h,
                                                  pad_w,
                                                  pad_h_end,
                                                  pad_w_end,
                                                  fuse_activation,
                                                  output);
            return kernel_activation_is_fused(fuse_activation);
        }

        static void AvgPool2dForwardImpl(const Tensor& input,
                                         int pool_h,
                                         int pool_w,
                                         int stride,
                                         int pad_h,
                                         int pad_w,
                                         int pad_h_end,
                                         int pad_w_end,
                                         Tensor& output)
        {
            if constexpr (NkAcceleratedKernel<LayerFast>)
            {
                if (pad_h_end == pad_h && pad_w_end == pad_w && pool_h == pool_w)
                {
                    if (LayerFast::TryAvgPool2dForward(
                            input, pool_h, stride, pad_h, pad_w, output))
                        return;
                }
            }
            ReferenceKernel::AvgPool2dForwardImpl(
                input, pool_h, pool_w, stride, pad_h, pad_w, pad_h_end, pad_w_end, output);
        }

        static void BatchNorm2dForwardImpl(const Tensor& input,
                                           const float* scale,
                                           const float* bias,
                                           int channels,
                                           Tensor& output)
        {
            if constexpr (NkAcceleratedKernel<LayerFast>)
            {
                if (!LayerFast::TryBatchNorm2dForward(input, scale, bias, channels, output))
                    ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
            }
            else if constexpr (NkAcceleratedKernel<VectorFast>)
            {
                if (!VectorFast::TryBatchNorm2dForward(input, scale, bias, channels, output))
                    ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
            }
            else
                ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
        }

        static void LayerNorm2dForwardImpl(const Tensor& input,
                                           const float* weight,
                                           const float* bias,
                                           int channels,
                                           float eps,
                                           Tensor& output)
        {
            if constexpr (NkAcceleratedKernel<VectorFast>)
            {
                if (VectorFast::TryLayerNorm2dForward(input, weight, bias, channels, eps, output))
                    return;
            }
            ReferenceKernel::LayerNorm2dForwardImpl(input, weight, bias, channels, eps, output);
        }

        static bool FullyConnectedWithBiasImpl(const Tensor& input,
                                               const Tensor& weights,
                                               const Tensor& bias,
                                               NetkitKernelActivation fuse_activation,
                                               Tensor& output)
        {
            return TryFullyConnected<LayerFast, VectorFast>(input, weights, bias, fuse_activation, output);
        }
    };

} // namespace detail
