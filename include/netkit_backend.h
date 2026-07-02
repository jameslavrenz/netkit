/*
 * netkit_backend.h — optional kernel backends (reference vs CMSIS-NN / CMSIS-DSP).
 *
 * Enable with Makefile:
 *   NETKIT_CMSIS_NN=1  -> NETKIT_USE_CMSIS_NN
 *   NETKIT_CMSIS_DSP=1 -> NETKIT_USE_CMSIS_DSP
 */
#ifndef NETKIT_BACKEND_H
#define NETKIT_BACKEND_H

#include "tensor.hpp"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NetkitBackendActivation
{
    NETKIT_BACKEND_ACT_NONE = 0,
    NETKIT_BACKEND_ACT_RELU,
    NETKIT_BACKEND_ACT_SIGMOID,
    NETKIT_BACKEND_ACT_TANH,
    NETKIT_BACKEND_ACT_LEAKY_RELU,
    NETKIT_BACKEND_ACT_RELU6,
    NETKIT_BACKEND_ACT_SOFTMAX,
} NetkitBackendActivation;

static inline int netkit_activation_is_fused(NetkitBackendActivation activation)
{
    return activation == NETKIT_BACKEND_ACT_RELU || activation == NETKIT_BACKEND_ACT_RELU6;
}

#if defined(NETKIT_USE_CMSIS_NN) && NETKIT_USE_CMSIS_NN

/* Returns 1 on success, 0 to fall back to the reference implementation. */
int netkit_cmsis_conv2d_forward(const Tensor* input,
                                const float* weights,
                                const float* bias,
                                int kernel_size,
                                int stride,
                                int in_channels,
                                int out_channels,
                                NetkitBackendActivation fuse_activation,
                                Tensor* output);

int netkit_cmsis_max_pool2d_forward(const Tensor* input,
                                    int pool_size,
                                    int stride,
                                    NetkitBackendActivation fuse_activation,
                                    Tensor* output);

int netkit_cmsis_fully_connected_forward(const Tensor* input,
                                         const Tensor* kernel,
                                         const Tensor* bias,
                                         NetkitBackendActivation fuse_activation,
                                         Tensor* output);

int netkit_cmsis_activation_forward(const Tensor* input,
                                    Tensor* output,
                                    NetkitBackendActivation activation,
                                    float leaky_alpha);

int netkit_cmsis_softmax_forward(const Tensor* input, Tensor* output);

int netkit_cmsis_nn_add_f32(const Tensor* a, const Tensor* b, Tensor* c);

#else

static inline int netkit_cmsis_conv2d_forward(const Tensor*,
                                              const float*,
                                              const float*,
                                              int,
                                              int,
                                              int,
                                              int,
                                              NetkitBackendActivation,
                                              Tensor*)
{
    return 0;
}

static inline int netkit_cmsis_max_pool2d_forward(const Tensor*, int, int, NetkitBackendActivation, Tensor*)
{
    return 0;
}

static inline int netkit_cmsis_fully_connected_forward(const Tensor*,
                                                       const Tensor*,
                                                       const Tensor*,
                                                       NetkitBackendActivation,
                                                       Tensor*)
{
    return 0;
}

static inline int netkit_cmsis_activation_forward(const Tensor*,
                                                  Tensor*,
                                                  NetkitBackendActivation,
                                                  float)
{
    return 0;
}

static inline int netkit_cmsis_softmax_forward(const Tensor*, Tensor*)
{
    return 0;
}

static inline int netkit_cmsis_nn_add_f32(const Tensor* a, const Tensor* b, Tensor* c)
{
    (void)a;
    (void)b;
    (void)c;
    return 0;
}

#endif

#if defined(NETKIT_USE_CMSIS_DSP) && NETKIT_USE_CMSIS_DSP

int netkit_cmsis_dsp_add_f32(const Tensor* a, const Tensor* b, Tensor* c);
int netkit_cmsis_dsp_mult_f32(const Tensor* a, const Tensor* b, Tensor* c);
int netkit_cmsis_dsp_scale_f32(const Tensor* a, float scalar, Tensor* c);
int netkit_cmsis_dsp_mat_mul_f32(const Tensor* a, const Tensor* b, Tensor* c);
int netkit_cmsis_dsp_clip_f32(const Tensor* input, Tensor* output, float low, float high);
int netkit_cmsis_dsp_fully_connected_forward(const Tensor* input,
                                             const Tensor* kernel,
                                             const Tensor* bias,
                                             Tensor* output);

#else

static inline int netkit_cmsis_dsp_add_f32(const Tensor* a, const Tensor* b, Tensor* c)
{
    (void)a;
    (void)b;
    (void)c;
    return 0;
}

static inline int netkit_cmsis_dsp_mult_f32(const Tensor* a, const Tensor* b, Tensor* c)
{
    (void)a;
    (void)b;
    (void)c;
    return 0;
}

static inline int netkit_cmsis_dsp_scale_f32(const Tensor* a, float scalar, Tensor* c)
{
    (void)a;
    (void)scalar;
    (void)c;
    return 0;
}

static inline int netkit_cmsis_dsp_mat_mul_f32(const Tensor* a, const Tensor* b, Tensor* c)
{
    (void)a;
    (void)b;
    (void)c;
    return 0;
}

static inline int netkit_cmsis_dsp_clip_f32(const Tensor* input, Tensor* output, float low, float high)
{
    (void)input;
    (void)output;
    (void)low;
    (void)high;
    return 0;
}

static inline int netkit_cmsis_dsp_fully_connected_forward(const Tensor* input,
                                                         const Tensor* kernel,
                                                         const Tensor* bias,
                                                         Tensor* output)
{
    (void)input;
    (void)kernel;
    (void)bias;
    (void)output;
    return 0;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* NETKIT_BACKEND_H */
