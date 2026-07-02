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
    constexpr bool IsReferenceKernel = std::is_same_v<T, ReferenceKernel>;

    template<typename Fast, typename Reference = ReferenceKernel>
    void TryVectorMul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if constexpr (!IsReferenceKernel<Fast>)
        {
            if (!Fast::TryMul(a, b, c))
                Reference::MulImpl(a, b, c);
        }
        else
            Reference::MulImpl(a, b, c);
    }

    template<typename Fast, typename Reference = ReferenceKernel>
    void TryVectorMulScalar(const Tensor& a, float scalar, Tensor& c)
    {
        if constexpr (!IsReferenceKernel<Fast>)
        {
            if (!Fast::TryMulScalar(a, scalar, c))
                Reference::MulScalarImpl(a, scalar, c);
        }
        else
            Reference::MulScalarImpl(a, scalar, c);
    }

    template<typename Fast, typename Reference = ReferenceKernel>
    void TryVectorMatMul(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if constexpr (!IsReferenceKernel<Fast>)
        {
            if (!Fast::TryMatMul(a, b, c))
                Reference::MatMulImpl(a, b, c);
        }
        else
            Reference::MatMulImpl(a, b, c);
    }

    template<typename LayerFast, typename VectorFast>
    void TryMatAdd2D(const Tensor& a, const Tensor& b, Tensor& c)
    {
        if constexpr (!IsReferenceKernel<LayerFast>)
        {
            if (!LayerFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddImpl(a, b, c);
        }
        else if constexpr (!IsReferenceKernel<VectorFast>)
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
        if constexpr (!IsReferenceKernel<LayerFast>)
        {
            if (!LayerFast::TryMatAdd(a, b, c))
                ReferenceKernel::MatAddNDImpl(a, b, c);
        }
        else if constexpr (!IsReferenceKernel<VectorFast>)
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
        if constexpr (!IsReferenceKernel<LayerFast>)
        {
            if (!LayerFast::TryActivationForward(a, c, activation, alpha))
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
        }
        else if constexpr (!IsReferenceKernel<VectorFast>)
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
            else if (activation == NetkitKernelActivation::LeakyReLU)
                ReferenceKernel::LeakyReLUImpl(a, c, alpha);
            else if (activation == NetkitKernelActivation::Sigmoid)
                ReferenceKernel::SigmoidImpl(a, c);
            else if (activation == NetkitKernelActivation::Tanh)
                ReferenceKernel::TanhImpl(a, c);
        }
        else
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
        if constexpr (!IsReferenceKernel<LayerFast>)
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
                                            in_channels,
                                            out_channels,
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
        if constexpr (!IsReferenceKernel<LayerFast>)
        {
            if (LayerFast::TryFullyConnectedWithBias(input, weights, bias, fuse_activation, output))
                return true;
        }
        else if constexpr (!IsReferenceKernel<VectorFast>)
        {
            if (VectorFast::TryFullyConnectedWithBias(input, weights, bias, output))
                return false;
        }
        return ReferenceKernel::FullyConnectedWithBiasImpl(input, weights, bias, fuse_activation, output);
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
            if constexpr (!IsReferenceKernel<LayerFast>)
            {
                if (!LayerFast::TrySoftmaxForward(a, c))
                    ReferenceKernel::SoftmaxImpl(a, c);
            }
            else
                ReferenceKernel::SoftmaxImpl(a, c);
        }

        static bool Conv2dForwardImpl(const Tensor& input,
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
            return TryLayerConv<LayerFast>(input,
                                             weights,
                                             bias,
                                             kernel_size,
                                             stride,
                                             pad_h,
                                             pad_w,
                                             in_channels,
                                             out_channels,
                                             fuse_activation,
                                             output);
        }

        static void MaxPool2dForwardImpl(const Tensor& input, int pool_size, int stride, Tensor& output)
        {
            if constexpr (!IsReferenceKernel<LayerFast>)
            {
                if (!LayerFast::TryMaxPool2dForward(
                        input, pool_size, stride, NetkitKernelActivation::None, output))
                    ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, output);
            }
            else
                ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, output);
        }

        static void AvgPool2dForwardImpl(const Tensor& input, int pool_size, int stride, Tensor& output)
        {
            if constexpr (!IsReferenceKernel<LayerFast>)
            {
                if (!LayerFast::TryAvgPool2dForward(input, pool_size, stride, output))
                    ReferenceKernel::AvgPool2dForwardImpl(input, pool_size, stride, output);
            }
            else
                ReferenceKernel::AvgPool2dForwardImpl(input, pool_size, stride, output);
        }

        static void BatchNorm2dForwardImpl(const Tensor& input,
                                           const float* scale,
                                           const float* bias,
                                           int channels,
                                           Tensor& output)
        {
            if constexpr (!IsReferenceKernel<LayerFast>)
            {
                if (!LayerFast::TryBatchNorm2dForward(input, scale, bias, channels, output))
                    ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
            }
            else if constexpr (!IsReferenceKernel<VectorFast>)
            {
                if (!VectorFast::TryBatchNorm2dForward(input, scale, bias, channels, output))
                    ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
            }
            else
                ReferenceKernel::BatchNorm2dForwardImpl(input, scale, bias, channels, output);
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
