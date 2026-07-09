# Host compile/link flags for fair comparison with TensorFlow Lite / LiteRT (MPU).
#
# Distinct from tflm_host_flags.mk, which mirrors TFLM Micro (-O2 +
# TF_LITE_DISABLE_X86_NEON) for MNIST/TFLM apples-to-apples runs.
#
# LiteRT wheels ship Bazel darwin_*-opt / Release builds with SIMD enabled.
# Match that profile for ImageNet / MPU host benches:
#   -O3 -DNDEBUG, no TF_LITE_DISABLE_X86_NEON, no TF_LITE_STATIC_MEMORY.
#
# Variable names reuse the TFLM_* namespace so bench.mk can switch profiles
# without duplicating recipes.
#
# Include from benchmark/netkit/bench.mk when BENCH_FLAG_PROFILE=tflite.

TFLM_HOST_CC := cc
TFLM_HOST_CXX := c++
TFLM_HOST_AR := ar

TFLM_CC_WARNINGS := \
  -Wsign-compare \
  -Wdouble-promotion \
  -Wunused-variable \
  -Wunused-function \
  -Wswitch \
  -Wvla \
  -Wall \
  -Wextra \
  -Wmissing-field-initializers \
  -Wstrict-aliasing \
  -Wno-unused-parameter

# Release-style host flags (LiteRT/XNNPACK opt). Keep -fno-rtti/-fno-exceptions
# because netkit's runtime does not use them; LiteRT's interpreter may differ,
# but the timed path is XNNPACK / netkit kernels, not the Python wrapper.
TFLM_CXXFLAGS := \
  -fno-rtti \
  -fno-exceptions \
  -fno-threadsafe-statics \
  -Wnon-virtual-dtor \
  -ffunction-sections \
  -fdata-sections \
  -fmessage-length=0 \
  $(TFLM_CC_WARNINGS) \
  -std=c++20

# Match CMake/Bazel Release for the MPU/desktop path (not TFLM's -O2).
TFLM_CORE_OPT := -O3 -DNDEBUG
TFLM_KERNEL_OPT := -O3 -DNDEBUG

TFLM_LDFLAGS := -lm

NETKIT_IM2COL ?= 0
NETKIT_LOOP_UNROLL ?= 0
NETKIT_BENCH_CPPFLAGS := \
  -DNETKIT_TARGET_CPU=1 \
  -DNETKIT_IM2COL=$(NETKIT_IM2COL) \
  -DNETKIT_LOOP_UNROLL=$(NETKIT_LOOP_UNROLL)

TFLM_BENCH_INCLUDES := -I$(ROOT)/include -I$(SHARED_GEN) -I../common

TFLM_BENCH_CORE_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_CORE_OPT) $(NETKIT_BENCH_CPPFLAGS) $(TFLM_BENCH_INCLUDES)
TFLM_BENCH_KERNEL_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(NETKIT_BENCH_CPPFLAGS) -I$(ROOT)/include
