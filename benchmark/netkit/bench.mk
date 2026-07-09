# Parameterized netkit MNIST benchmark build (included from Makefile).
#
# Required variables from caller:
#   BACKEND          — "reference", "loop-unroll", "cmsis-dsp", or "xnnpack" (summary label)
#   CMSIS_DSP        — 0 or 1
#   XNNPACK          — 0 or 1 (cpu/mpu LayerFast; requires ./tools/fetch_xnnpack.sh)
#   NETKIT_LOOP_UNROLL — 0 or 1 (reference kernels only; set via tflm_host_flags.mk)
#   BENCH_OBJDIR     — object directory for this variant
#   MLP_BENCH        — MLP binary name
#   CNN_BENCH        — CNN binary name
#   MLP_MAIN_OBJ     — compiled main.o path
#   CNN_MAIN_OBJ     — compiled cnn main.o path
#   BENCH_LIB        — static library name

ROOT := $(abspath ../..)
SHARED_GEN := ../tflm/generated
# Flag profile:
#   tflm   — match TFLM Micro host (-O2, TF_LITE_DISABLE_X86_NEON); MNIST/TFLM compare
#   tflite — match TF Lite / LiteRT MPU host (-O3 -DNDEBUG, SIMD on); ImageNet compare
BENCH_FLAG_PROFILE ?= tflm
ifeq ($(BENCH_FLAG_PROFILE),tflite)
  include ../common/tflite_host_flags.mk
else
  include ../common/tflm_host_flags.mk
endif

CMSIS_DSP_DIR := $(ROOT)/third_party/CMSIS-DSP
XNNPACK_DIR := $(ROOT)/third_party/XNNPACK
XNNPACK_LIB_DIR := $(XNNPACK_DIR)/netkit_lib
XNNPACK ?= 0

BENCH_RUNTIME_SOURCES := \
  src/arena.cpp \
  src/tensor_factory.cpp \
  src/tensor_access.cpp \
  src/reference_kernel.cpp \
  src/kernel_workspace.cpp \
  src/cmsis_dsp_util.cpp \
  src/cmsis_buffer_size.cpp \
  src/ops.cpp \
  src/conv2d.cpp \
  src/conv2d_layout.cpp \
  src/conv_dispatch.cpp \
  src/conv1x1_kernel.cpp \
  src/conv_depthwise_kernel.cpp \
  src/conv_direct_kernel.cpp \
  src/im2col_partial.cpp \
  src/im2col_full.cpp \
  src/depthwise_conv2d.cpp \
  src/convnextv2_block.cpp \
  src/mobilenetv4_uib.cpp \
  src/resnet_basic_block.cpp \
  src/yolox_decoupled_head.cpp \
  src/mlp.cpp \
  src/cnn.cpp \
  src/layer_ops/nk_op_conv2d.cpp \
  src/layer_ops/nk_op_depthwise_conv2d.cpp \
  src/layer_ops/nk_op_convnextv2_block.cpp \
  src/layer_ops/nk_op_mobilenetv4_uib.cpp \
  src/layer_ops/nk_op_yolox_decoupled_head.cpp \
  src/layer_ops/nk_op_resnet_basic_block.cpp \
  src/layer_ops/nk_op_layernorm2d.cpp \
  src/layer_ops/nk_op_max_pool2d.cpp \
  src/layer_ops/nk_op_avg_pool2d.cpp \
  src/layer_ops/nk_op_batch_norm2d.cpp \
  src/layer_ops/nk_op_flatten.cpp \
  src/layer_ops/nk_op_dense.cpp \
  src/ops_resolver.cpp \
  src/ops_resolver_default.cpp \
  src/nk_format.cpp \
  src/nk_loader.cpp \
  src/netkit_api.cpp \
  src/quant_ops.cpp \
  src/quant_softmax_s8.cpp \
  src/quant_trace.cpp \
  src/cmsis_nn_quant_backend.cpp \
  src/cmsis_quant_plan.cpp \
  src/cnn_quant.cpp \
  src/xnnpack_quant_backend.cpp

ifeq ($(CMSIS_DSP),1)
  ifeq ($(wildcard $(CMSIS_DSP_DIR)/Include/arm_math.h),)
    $(error CMSIS-DSP not found at $(CMSIS_DSP_DIR) — run ./tools/fetch_cmsis_dsp.sh)
  endif
  BENCH_RUNTIME_SOURCES += src/cmsis_dsp_backend.cpp
  NETKIT_BENCH_VARIANT_CPPFLAGS := \
    -DNETKIT_USE_CMSIS_DSP=1 \
    -D__GNUC_PYTHON__ \
    -DARM_MATH_LOOPUNROLL
  CMSIS_DSP_SOURCES := \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_add_f32.c \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_mult_f32.c \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_offset_f32.c \
    $(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_clip_f32.c \
    $(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_mean_f32.c \
    $(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_var_f32.c \
    $(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_f32.c \
    $(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_q7.c \
    $(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_init_f32.c \
    $(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
    $(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_mult_f32.c \
    $(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_f32.c \
    $(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_q7.c
  CMSIS_DSP_OBJS := $(patsubst $(CMSIS_DSP_DIR)/%.c,$(BENCH_OBJDIR)/cmsis_dsp/%.o,$(CMSIS_DSP_SOURCES))
  CMSIS_DSP_CFLAGS := -std=c11 $(TFLM_KERNEL_OPT) \
    -I$(CMSIS_DSP_DIR)/Include \
    -I$(CMSIS_DSP_DIR)/PrivateInclude \
    $(NETKIT_BENCH_VARIANT_CPPFLAGS)
  CMSIS_DSP_INCLUDES := -I$(CMSIS_DSP_DIR)/Include -I$(CMSIS_DSP_DIR)/PrivateInclude
else
  NETKIT_BENCH_VARIANT_CPPFLAGS :=
  CMSIS_DSP_OBJS :=
  CMSIS_DSP_INCLUDES :=
endif

XNNPACK_LDFLAGS :=
ifeq ($(XNNPACK),1)
  ifeq ($(wildcard $(XNNPACK_DIR)/include/xnnpack.h),)
    $(error XNNPACK not found at $(XNNPACK_DIR) — run ./tools/fetch_xnnpack.sh)
  endif
  ifeq ($(wildcard $(XNNPACK_LIB_DIR)/libXNNPACK.a),)
    $(error libXNNPACK.a missing — run ./tools/fetch_xnnpack.sh)
  endif
  BENCH_RUNTIME_SOURCES += src/xnnpack_backend.cpp
  NETKIT_BENCH_VARIANT_CPPFLAGS += -DNETKIT_USE_XNNPACK=1 -I$(XNNPACK_DIR)/include -I$(XNNPACK_DIR)/netkit_include
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    XNNPACK_LDFLAGS += -Wl,-force_load,$(XNNPACK_LIB_DIR)/libXNNPACK.a
  else
    XNNPACK_LDFLAGS += -Wl,--whole-archive -L$(XNNPACK_LIB_DIR) -lXNNPACK -Wl,--no-whole-archive
  endif
  XNNPACK_LDFLAGS += -L$(XNNPACK_LIB_DIR)
  ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libxnnpack-microkernels-prod.a),)
    XNNPACK_LDFLAGS += -lxnnpack-microkernels-prod
  else ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libxnnpack-microkernels-all.a),)
    XNNPACK_LDFLAGS += -lxnnpack-microkernels-all
  endif
  ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libkleidiai.a),)
    XNNPACK_LDFLAGS += -lkleidiai
  endif
  ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libpthreadpool.a),)
    XNNPACK_LDFLAGS += -lpthreadpool
  endif
  ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libcpuinfo.a),)
    XNNPACK_LDFLAGS += -lcpuinfo
  endif
  ifneq ($(wildcard $(XNNPACK_LIB_DIR)/libfxdiv.a),)
    XNNPACK_LDFLAGS += -lfxdiv
  endif
  XNNPACK_LDFLAGS += -lpthread -lc++
endif

BENCH_KERNEL_CPPFLAGS := $(NETKIT_BENCH_CPPFLAGS) $(NETKIT_BENCH_VARIANT_CPPFLAGS)
BENCH_KERNEL_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(BENCH_KERNEL_CPPFLAGS) $(CMSIS_DSP_INCLUDES) -I$(ROOT)/include
BENCH_CORE_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_CORE_OPT) $(BENCH_KERNEL_CPPFLAGS) $(TFLM_BENCH_INCLUDES)
BENCH_MAIN_CPPFLAGS := -DNETKIT_BENCH_BACKEND=\"$(BACKEND)\"

MLP_SRC := src/main.cc
CNN_SRC := src/mnist_cnn_main.cc
MNV4_SRC := src/mobilenetv4_main.cc
MNV4_IMAGENET_SRC := src/mobilenetv4_imagenet_main.cc
MNV4_IMAGENET_INT8_SRC := src/mobilenetv4_imagenet_int8_main.cc
MLP_PROFILE_SRC := src/mnist_mlp_profile_main.cc
CNN_PROFILE_SRC := src/mnist_cnn_profile_main.cc
MLP_IMAGES_CC := $(SHARED_GEN)/mnist_test_images.cc
CNN_IMAGES_CC := $(SHARED_GEN)/mnist_cnn_test_images.cc
MNV4_INT8_IMAGES_CC := $(SHARED_GEN)/mobilenetv4_netkit_int8_test_images.cc
IMAGENET_IMAGES_CC := $(SHARED_GEN)/imagenet_mnv4_test_images.cc
IMAGENET_INT8_IMAGES_CC := $(SHARED_GEN)/imagenet_mnv4_netkit_int8_test_images.cc
MLP_IMAGES_OBJ := $(SHARED_GEN)/mnist_test_images.o
CNN_IMAGES_OBJ := $(SHARED_GEN)/mnist_cnn_test_images.o
MNV4_INT8_IMAGES_OBJ := $(SHARED_GEN)/mobilenetv4_netkit_int8_test_images.o
IMAGENET_IMAGES_OBJ := $(SHARED_GEN)/imagenet_mnv4_test_images.o
IMAGENET_INT8_IMAGES_OBJ := $(SHARED_GEN)/imagenet_mnv4_netkit_int8_test_images.o

CNN_PROFILE_BENCH ?= mnist_cnn_profile_bench
CNN_PROFILE_MAIN_OBJ ?= src/mnist_cnn_profile_main.o
MLP_PROFILE_BENCH ?= mnist_mlp_profile_bench
MLP_PROFILE_MAIN_OBJ ?= src/mnist_mlp_profile_main.o
MNV4_BENCH ?= mobilenetv4_bench
MNV4_MAIN_OBJ ?= src/mobilenetv4_main.o
MNV4_IMAGENET_BENCH ?= mobilenetv4_imagenet_bench
MNV4_IMAGENET_MAIN_OBJ ?= src/mobilenetv4_imagenet_main.o
MNV4_IMAGENET_INT8_BENCH ?= mobilenetv4_imagenet_int8_bench
MNV4_IMAGENET_INT8_MAIN_OBJ ?= src/mobilenetv4_imagenet_int8_main.o

BENCH_LIB_OBJS := $(addprefix $(BENCH_OBJDIR)/,$(BENCH_RUNTIME_SOURCES:.cpp=.o))

.PHONY: build-mlp build-cnn build-cnn-profile build-mlp-profile build-lib build-mobilenetv4 \
  build-mobilenetv4-imagenet build-mobilenetv4-imagenet-int8 \
  run-mobilenetv4 run-mobilenetv4-imagenet run-mobilenetv4-imagenet-int8

build-lib: $(BENCH_LIB)
build-mlp: $(MLP_BENCH)
build-cnn: $(CNN_BENCH)
build-cnn-profile: $(CNN_PROFILE_BENCH)
build-mlp-profile: $(MLP_PROFILE_BENCH)
build-mobilenetv4: $(MNV4_BENCH)
build-mobilenetv4-imagenet: $(MNV4_IMAGENET_BENCH)
build-mobilenetv4-imagenet-int8: $(MNV4_IMAGENET_INT8_BENCH)

$(MLP_IMAGES_CC) $(CNN_IMAGES_CC):
	$(MAKE) -C ../tflm export-assets

$(IMAGENET_IMAGES_CC):
	$(MAKE) -C ../tflm export-imagenet-mnv4-images

$(MNV4_INT8_IMAGES_CC):
	@test -f $(ROOT)/models/mobilenetv4_small_int8.nk || \
	  (cd $(ROOT) && python3 tools/write_mobilenetv4_small_int8.py)
	@cd ../tflm && python3 tools/export_mobilenetv4_int8_test_images.py --quant-source nk

$(IMAGENET_INT8_IMAGES_CC):
	@test -f $(ROOT)/models/mobilenetv4_imagenet_int8.nk || \
	  (cd $(ROOT) && python3 tools/write_mobilenetv4_imagenet_int8.py)
	@test -d $(SHARED_GEN)/imagenet_sample_cache || $(MAKE) -C ../tflm export-imagenet-mnv4-images
	@cd ../tflm && python3 tools/export_imagenet_mnv4_int8_test_images.py --quant-source nk

$(BENCH_OBJDIR)/%.o: $(ROOT)/%.cpp
	@mkdir -p $(dir $@)
	$(TFLM_HOST_CXX) $(BENCH_KERNEL_CXXFLAGS) -c $< -o $@

$(BENCH_OBJDIR)/cmsis_dsp/%.o: $(CMSIS_DSP_DIR)/%.c
	@mkdir -p $(dir $@)
	$(TFLM_HOST_CC) $(CMSIS_DSP_CFLAGS) -c $< -o $@

$(BENCH_LIB): $(BENCH_LIB_OBJS) $(CMSIS_DSP_OBJS)
	$(TFLM_HOST_AR) rcs $@ $^

# Link with the same opt tier as compile (LTO-free; keeps profile consistent).
BENCH_LINK_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(BENCH_KERNEL_CPPFLAGS) $(TFLM_BENCH_INCLUDES)

$(MLP_BENCH): $(BENCH_LIB) $(MLP_MAIN_OBJ) $(MLP_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(MLP_MAIN_OBJ) $(MLP_IMAGES_OBJ) $(BENCH_LIB) $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(CNN_BENCH): $(BENCH_LIB) $(CNN_MAIN_OBJ) $(CNN_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(CNN_MAIN_OBJ) $(CNN_IMAGES_OBJ) $(BENCH_LIB) $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

# MobileNetV4-small: float uses MNIST CNN images; int8 also links netkit-scaled int8 fixtures.
$(MNV4_BENCH): $(BENCH_LIB) $(MNV4_MAIN_OBJ) $(CNN_IMAGES_OBJ) $(MNV4_INT8_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(MNV4_MAIN_OBJ) $(CNN_IMAGES_OBJ) $(MNV4_INT8_IMAGES_OBJ) $(BENCH_LIB) \
	  $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(MNV4_IMAGENET_BENCH): $(BENCH_LIB) $(MNV4_IMAGENET_MAIN_OBJ) $(IMAGENET_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(MNV4_IMAGENET_MAIN_OBJ) $(IMAGENET_IMAGES_OBJ) $(BENCH_LIB) $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(MNV4_IMAGENET_INT8_BENCH): $(BENCH_LIB) $(MNV4_IMAGENET_INT8_MAIN_OBJ) $(IMAGENET_INT8_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(MNV4_IMAGENET_INT8_MAIN_OBJ) $(IMAGENET_INT8_IMAGES_OBJ) $(BENCH_LIB) \
	  $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(CNN_PROFILE_BENCH): $(BENCH_LIB) $(CNN_PROFILE_MAIN_OBJ) $(CNN_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(CNN_PROFILE_MAIN_OBJ) $(CNN_IMAGES_OBJ) $(BENCH_LIB) $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(MLP_PROFILE_BENCH): $(BENCH_LIB) $(MLP_PROFILE_MAIN_OBJ) $(MLP_IMAGES_OBJ)
	$(TFLM_HOST_CXX) $(BENCH_LINK_CXXFLAGS) \
	  -o $@ $(MLP_PROFILE_MAIN_OBJ) $(MLP_IMAGES_OBJ) $(BENCH_LIB) $(TFLM_LDFLAGS) $(XNNPACK_LDFLAGS)

$(MLP_MAIN_OBJ): $(MLP_SRC) $(MLP_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(CNN_MAIN_OBJ): $(CNN_SRC) $(CNN_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(MNV4_MAIN_OBJ): $(MNV4_SRC) $(CNN_IMAGES_CC) $(MNV4_INT8_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(MNV4_IMAGENET_MAIN_OBJ): $(MNV4_IMAGENET_SRC) $(IMAGENET_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(MNV4_IMAGENET_INT8_MAIN_OBJ): $(MNV4_IMAGENET_INT8_SRC) $(IMAGENET_INT8_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(CNN_PROFILE_MAIN_OBJ): $(CNN_PROFILE_SRC) $(CNN_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(MLP_PROFILE_MAIN_OBJ): $(MLP_PROFILE_SRC) $(MLP_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) $(BENCH_MAIN_CPPFLAGS) -c $< -o $@

$(MLP_IMAGES_OBJ): $(MLP_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(CNN_IMAGES_OBJ): $(CNN_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(MNV4_INT8_IMAGES_OBJ): $(MNV4_INT8_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(IMAGENET_IMAGES_OBJ): $(IMAGENET_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(IMAGENET_INT8_IMAGES_OBJ): $(IMAGENET_INT8_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

run-mlp: $(MLP_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(MLP_BENCH) models/mnist_mlp.nk

run-cnn: $(CNN_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(CNN_BENCH) models/mnist_cnn.nk

run-mobilenetv4: $(MNV4_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(MNV4_BENCH) models/mobilenetv4_small.nk

run-mobilenetv4-int8: $(MNV4_BENCH)
	@test -f $(ROOT)/models/mobilenetv4_small_int8.nk || \
	  (cd $(ROOT) && python3 tools/write_mobilenetv4_small_int8.py)
	@cd $(ROOT) && ./benchmark/netkit/$(MNV4_BENCH) models/mobilenetv4_small_int8.nk

run-mobilenetv4-imagenet: $(MNV4_IMAGENET_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(MNV4_IMAGENET_BENCH) models/mobilenetv4_imagenet_f32.nk

run-mobilenetv4-imagenet-int8: $(MNV4_IMAGENET_INT8_BENCH)
	@test -f $(ROOT)/models/mobilenetv4_imagenet_int8.nk || \
	  (cd $(ROOT) && python3 tools/write_mobilenetv4_imagenet_int8.py)
	@cd $(ROOT) && ./benchmark/netkit/$(MNV4_IMAGENET_INT8_BENCH) models/mobilenetv4_imagenet_int8.nk

run-cnn-profile: $(CNN_PROFILE_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(CNN_PROFILE_BENCH) models/mnist_cnn.nk

run-mlp-profile: $(MLP_PROFILE_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(MLP_PROFILE_BENCH) models/mnist_mlp.nk
