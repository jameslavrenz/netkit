# NUCLEO-F446RE (STM32F446RET6) — MNIST CNN int8 benchmark firmware.
#
# Build profile (explicit CMSIS opt-in — not inferred from NETKIT_TARGET alone):
#   NETKIT_CMSIS_DSP=1, NETKIT_CMSIS_NN=1
#   NETKIT_TARGET=mcu, NETKIT_ARCH=CM4 (weights always flash/blob-backed)
#   Toolchain: boards/nucleo-f446re/mcu_tflm_toolchain.mk (TFLM microlite tiers:
#     CORE/KERNEL/THIRD_PARTY all -O2, -flto link)
#   Default: interpreter embed (--no-lower) — fair vs TFLM MicroInterpreter
#   Optional: NETKIT_LOWERED=1 for quant lowered deployment profile
#
# Usage:
#   make cmsis-init          # once, from repo root
#   make export-mnist-cnn-int8  # once, from repo root (if model missing)
#   make                     # build firmware.elf (interpreter embed)
#   ./scripts/flash.sh       # program via ST-Link (USB)
#   ./scripts/monitor.sh     # read UART results (ST-Link VCP)

NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../nucleo-f446re)

# netkit public headers use C++20 (std::span, constinit); TFLM board glue stays C++17.
MCU_CXX_STD := -std=c++20
MCU_BOARD_COMMON := $(BOARD_COMMON)

include $(BOARD_COMMON)/mcu_tflm_toolchain.mk

# 0 = interpreter embed (default, TFLM-fair). 1 = quant lowered call chain.
NETKIT_LOWERED ?= 0

BUILD_DIR := build
GEN_DIR := generated
LIB_DIR := $(BUILD_DIR)/lib
CMSIS_NN_DIR := $(NETKIT_ROOT)/third_party/CMSIS-NN
CMSIS_DSP_DIR := $(NETKIT_ROOT)/third_party/CMSIS-DSP

TARGET := mnist_cnn_int8_nucleo_f446re
ELF := $(BUILD_DIR)/$(TARGET).elf
BIN := $(BUILD_DIR)/$(TARGET).bin

LINKER_SCRIPT := $(BOARD_ROOT)/linker/STM32F446RETx_FLASH.ld

INCLUDES := \
	-I$(BOARD_COMMON)/include \
	-I$(NETKIT_ROOT)/include \
	-I$(GEN_DIR) \
	-I$(NETKIT_ROOT)/benchmark/tflm/generated \
	-I$(CMSIS_DSP_DIR)/Include \
	-I$(CMSIS_DSP_DIR)/PrivateInclude \
	-I$(CMSIS_NN_DIR)/Include \
	-I$(NETKIT_ROOT)/third_party/CMSIS-Core/CMSIS/Core/Include

NETKIT_QUANT_TRACE ?= 0
NETKIT_MCU_QUANT_ONLY ?= 1
# float Conv2D strategy (int8 path uses CMSIS-NN/QuantOps; kept for the float fallback): 0 = direct loops.
NETKIT_IM2COL ?= 0
# 1 = netkit reference int8 quant loops (QuantOps) instead of CMSIS-NN kernels.
NETKIT_REFERENCE_QUANT_LOOPS ?= 0

# Board firmware: CMSIS-DSP always on. CMSIS-NN on unless reference quant loops requested.
# Do not inherit CI/host NETKIT_CMSIS_*=0 from env.
override NETKIT_CMSIS_DSP := 1
ifeq ($(NETKIT_REFERENCE_QUANT_LOOPS),1)
override NETKIT_CMSIS_NN := 0
else
override NETKIT_CMSIS_NN := 1
endif

ifeq ($(NETKIT_QUANT_TRACE),1)
NETKIT_TRACE_SOURCES := $(NETKIT_ROOT)/src/quant_trace.cpp
else
NETKIT_TRACE_SOURCES :=
endif

NETKIT_CPPFLAGS := \
	-DNETKIT_TARGET_MCU=1 \
	-DNETKIT_MCU_QUANT_ONLY=$(NETKIT_MCU_QUANT_ONLY) \
	-DNETKIT_KERNELS_OPTIMIZED_FOR_SPEED=1 \
	$(if $(filter 1,$(NETKIT_CMSIS_DSP)),-DNETKIT_USE_CMSIS_DSP=1,) \
	$(if $(filter 1,$(NETKIT_CMSIS_NN)),-DNETKIT_USE_CMSIS_NN=1,) \
	-DNETKIT_QUANT_TRACE=$(NETKIT_QUANT_TRACE) \
	$(if $(filter 0,$(NETKIT_QUANT_TRACE)),,-DNETKIT_QUANT_TRACE_UART=1) \
	-DNETKIT_LOOP_UNROLL=0 \
	-DNETKIT_IM2COL=$(NETKIT_IM2COL) \
	-DNETKIT_REFERENCE_QUANT_LOOPS=$(NETKIT_REFERENCE_QUANT_LOOPS) \
	-DARM_MATH_CM4 \
	-DARM_MATH_LOOPUNROLL \
	-DARM_NN_ENABLE_F32=0

ifeq ($(NETKIT_MCU_QUANT_ONLY),1)
NETKIT_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/arena.cpp \
	$(NETKIT_ROOT)/src/tensor_factory.cpp \
	$(NETKIT_ROOT)/src/tensor_access.cpp \
	$(NETKIT_ROOT)/src/kernel_workspace.cpp \
	$(NETKIT_ROOT)/src/cmsis_buffer_size.cpp \
	$(NETKIT_ROOT)/src/quant_ops.cpp \
	$(NETKIT_ROOT)/src/quant_softmax_s8.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_util.cpp \
	$(NETKIT_ROOT)/src/cmsis_nn_quant_backend.cpp \
	$(NETKIT_ROOT)/src/cmsis_quant_plan.cpp \
	$(NETKIT_ROOT)/src/cnn_quant.cpp \
	$(NETKIT_ROOT)/src/cnn.cpp \
	$(NETKIT_ROOT)/src/conv2d_layout.cpp \
	$(NETKIT_ROOT)/src/ops_resolver.cpp \
	$(NETKIT_ROOT)/src/mobilenetv4_uib.cpp \
	$(NETKIT_ROOT)/src/resnet_basic_block.cpp \
	$(NETKIT_ROOT)/src/convnextv2_block.cpp \
	$(NETKIT_ROOT)/src/yolox_decoupled_head.cpp \
	$(NETKIT_TRACE_SOURCES) \
	$(NETKIT_ROOT)/src/nk_format.cpp \
	$(NETKIT_ROOT)/src/nk_loader.cpp
else
NETKIT_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/arena.cpp \
	$(NETKIT_ROOT)/src/tensor_factory.cpp \
	$(NETKIT_ROOT)/src/tensor_access.cpp \
	$(NETKIT_ROOT)/src/reference_kernel.cpp \
	$(NETKIT_ROOT)/src/kernel_workspace.cpp \
	$(NETKIT_ROOT)/src/cmsis_buffer_size.cpp \
	$(NETKIT_ROOT)/src/ops.cpp \
	$(NETKIT_ROOT)/src/conv2d.cpp \
	$(NETKIT_ROOT)/src/depthwise_conv2d.cpp \
	$(NETKIT_ROOT)/src/conv2d_layout.cpp \
	$(NETKIT_ROOT)/src/conv_dispatch.cpp \
	$(NETKIT_ROOT)/src/conv1x1_kernel.cpp \
	$(NETKIT_ROOT)/src/conv_depthwise_kernel.cpp \
	$(NETKIT_ROOT)/src/conv_direct_kernel.cpp \
	$(NETKIT_ROOT)/src/im2col_partial.cpp \
	$(NETKIT_ROOT)/src/im2col_full.cpp \
	$(NETKIT_ROOT)/src/convnextv2_block.cpp \
	$(NETKIT_ROOT)/src/mobilenetv4_uib.cpp \
	$(NETKIT_ROOT)/src/yolox_decoupled_head.cpp \
	$(NETKIT_ROOT)/src/resnet_basic_block.cpp \
	$(NETKIT_ROOT)/src/mlp.cpp \
	$(NETKIT_ROOT)/src/quant_ops.cpp \
	$(NETKIT_ROOT)/src/quant_softmax_s8.cpp \
	$(NETKIT_TRACE_SOURCES) \
	$(NETKIT_ROOT)/src/cmsis_nn_quant_backend.cpp \
	$(NETKIT_ROOT)/src/cmsis_quant_plan.cpp \
	$(NETKIT_ROOT)/src/cnn_quant.cpp \
	$(NETKIT_ROOT)/src/cnn.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_conv2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_depthwise_conv2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_convnextv2_block.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_mobilenetv4_uib.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_yolox_decoupled_head.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_resnet_basic_block.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_layernorm2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_max_pool2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_avg_pool2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_batch_norm2d.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_flatten.cpp \
	$(NETKIT_ROOT)/src/layer_ops/nk_op_dense.cpp \
	$(NETKIT_ROOT)/src/ops_resolver.cpp \
	$(NETKIT_ROOT)/src/ops_resolver_default.cpp \
	$(NETKIT_ROOT)/src/nk_format.cpp \
	$(NETKIT_ROOT)/src/nk_loader.cpp \
	$(NETKIT_ROOT)/src/netkit_api.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_backend.cpp \
	$(NETKIT_ROOT)/src/cmsis_nn_backend.cpp
endif

# Compiled at KERNEL_OPTIMIZATION_LEVEL (-O2), same tier as TFLM cmsis_nn kernels.
NETKIT_KERNEL_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/cmsis_nn_quant_backend.cpp \
	$(NETKIT_ROOT)/src/cmsis_quant_plan.cpp \
	$(NETKIT_ROOT)/src/cnn_quant.cpp \
	$(NETKIT_ROOT)/src/quant_softmax_s8.cpp \
	$(NETKIT_ROOT)/src/quant_ops.cpp

NETKIT_CORE_RUNTIME_SOURCES := $(filter-out $(NETKIT_KERNEL_RUNTIME_SOURCES),$(NETKIT_RUNTIME_SOURCES))

CMSIS_DSP_SOURCES := \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_add_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_offset_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_clip_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_mean_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_var_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_mult_f32.c

include $(NETKIT_ROOT)/third_party/cmsis_nn.mk

ifeq ($(NETKIT_MCU_QUANT_ONLY),1)
CMSIS_NN_CFLAGS := -std=c11 -O2 -DARM_NN_ENABLE_F32=0 -I$(CMSIS_NN_DIR)/Include $(NETKIT_ARCH_CFLAGS)
ifeq ($(NETKIT_REFERENCE_QUANT_LOOPS),1)
CMSIS_NN_SOURCES :=
else
CMSIS_NN_SOURCES := \
	$(CMSIS_NN_S8_SOURCES) \
	$(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_softmax_s8.c \
	$(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c
endif
ifeq ($(NETKIT_CMSIS_DSP),1)
CMSIS_DSP_SOURCES := \
	$(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_q7.c \
	$(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_q7.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_q7.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_offset_f32.c
else
CMSIS_DSP_SOURCES :=
endif
endif

BOARD_C_SOURCES := \
	$(BOARD_COMMON)/src/system_stm32f4xx.c \
	$(BOARD_COMMON)/src/uart.c \
	$(BOARD_COMMON)/src/dwt_time.c \
	$(BOARD_COMMON)/src/syscalls.c

BOARD_CXX_SOURCES := \
	src/main.cpp \
	$(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc

BOARD_ASM_SOURCES := $(BOARD_COMMON)/startup/startup_stm32f446xx.S

GEN_SOURCES := \
	$(GEN_DIR)/mnist_cnn_int8_aot.cpp

CFLAGS := $(MCU_CORE_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
CXXFLAGS := $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
ASFLAGS := $(MCU_ASFLAGS) $(MCU_DEBUG_FLAGS)
LDFLAGS := $(MCU_LDFLAGS) $(NETKIT_CPPFLAGS) -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

LIBNETKIT := $(LIB_DIR)/libnetkit_cnn_int8.a

.PHONY: all clean flash monitor run export-embed check-deps size

all: check-deps export-embed $(ELF) size

check-deps:
	$(MCU_ASSERT_GCC_TOOLCHAIN)
	@test -f $(CMSIS_DSP_DIR)/Include/arm_math.h || \
		(echo "Run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(CMSIS_NN_DIR)/Include/arm_nnfunctions.h || \
		(echo "Run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/models/mnist_cnn_int8.nk || \
		(echo "Missing models/mnist_cnn_int8.nk — run: make -C $(NETKIT_ROOT) export-mnist-cnn-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc || \
		(echo "Missing int8 test images — run: python3 benchmark/tflm/tools/export_int8_test_images.py" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NETKIT_ROOT)/models/mnist_cnn_int8.nk
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python python3 -m netkit aot models/mnist_cnn_int8.nk \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		$(if $(filter 1,$(NETKIT_LOWERED)),,--no-lower) --omit-final-softmax
	@touch $(GEN_DIR)/.embed_stamp

# Back-compat alias (prefer export-embed).
export-aot: export-embed

$(LIBNETKIT): $(NETKIT_RUNTIME_SOURCES) $(CMSIS_DSP_SOURCES) $(CMSIS_NN_SOURCES)
	@mkdir -p $(LIB_DIR)
	@rm -f $(LIB_DIR)/*.o
	@set -e; for src in $(CMSIS_DSP_SOURCES); do \
		obj=$(LIB_DIR)/cmsis_dsp_$$(basename $$src .c).o; \
		$(CC) $(MCU_THIRD_PARTY_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(CMSIS_NN_SOURCES); do \
		obj=$(LIB_DIR)/cmsis_nn_$$(basename $$src .c).o; \
		$(CC) $(MCU_THIRD_PARTY_KERNEL_CFLAGS) $(MCU_DEBUG_FLAGS) -I$(CMSIS_NN_DIR)/Include $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(NETKIT_CORE_RUNTIME_SOURCES); do \
		obj=$(LIB_DIR)/netkit_$$(basename $$src .cpp).o; \
		$(CXX) $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(NETKIT_KERNEL_RUNTIME_SOURCES); do \
		obj=$(LIB_DIR)/netkit_$$(basename $$src .cpp).o; \
		$(CXX) $(MCU_KERNEL_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS) -c $$src -o $$obj; \
	done
	$(AR) rcs $@ $(LIB_DIR)/*.o

OBJS := \
	$(BUILD_DIR)/startup.o \
	$(BUILD_DIR)/system_stm32f4xx.o \
	$(BUILD_DIR)/uart.o \
	$(BUILD_DIR)/dwt_time.o \
	$(BUILD_DIR)/syscalls.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/mnist_cnn_int8_test_images.o \
	$(BUILD_DIR)/mnist_cnn_int8_aot.o

$(BUILD_DIR)/startup.o: $(BOARD_ASM_SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/system_stm32f4xx.o: $(BOARD_COMMON)/src/system_stm32f4xx.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/uart.o: $(BOARD_COMMON)/src/uart.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/dwt_time.o: $(BOARD_COMMON)/src/dwt_time.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/syscalls.o: $(BOARD_COMMON)/src/syscalls.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -fno-lto -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_cnn_int8_test_images.o: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_cnn_int8_aot.o: $(GEN_DIR)/mnist_cnn_int8_aot.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LIBNETKIT)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIBNETKIT) -lm -lc -lgcc -lstdc++ -lsupc++ -o $@
	$(OBJCOPY) -O binary $@ $(BIN)

size: $(ELF)
	$(SIZE) --format=berkeley $(ELF)
	@echo "Flash budget: 512 KiB — see .text + .data load size above"

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(GEN_DIR)/.embed_stamp $(GEN_DIR)/.aot_stamp

flash: $(ELF)
	./scripts/flash.sh $(ELF)

monitor:
	./scripts/monitor.sh

run: flash monitor
