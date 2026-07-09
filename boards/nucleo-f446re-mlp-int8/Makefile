# NUCLEO-F446RE (STM32F446RET6) — MNIST MLP int8 benchmark firmware.
#
# Usage:
#   make -C ../.. export-mnist-mlp-int8   # once
#   make NETKIT_REFERENCE_QUANT_LOOPS=0   # CMSIS-NN (default)
#   make NETKIT_REFERENCE_QUANT_LOOPS=1   # reference QuantOps loops
#   ./scripts/flash.sh

NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../nucleo-f446re)

MCU_CXX_STD := -std=c++20
MCU_BOARD_COMMON := $(BOARD_COMMON)

include $(BOARD_COMMON)/mcu_tflm_toolchain.mk

BUILD_DIR := build
GEN_DIR := generated
LIB_DIR := $(BUILD_DIR)/lib
CMSIS_NN_DIR := $(NETKIT_ROOT)/third_party/CMSIS-NN
CMSIS_DSP_DIR := $(NETKIT_ROOT)/third_party/CMSIS-DSP

TARGET := mnist_mlp_int8_nucleo_f446re
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

NETKIT_WEIGHTS_IN_RAM ?= 0
NETKIT_REFERENCE_QUANT_LOOPS ?= 0

override NETKIT_CMSIS_DSP := 1
ifeq ($(NETKIT_REFERENCE_QUANT_LOOPS),1)
override NETKIT_CMSIS_NN := 0
else
override NETKIT_CMSIS_NN := 1
endif

NETKIT_CPPFLAGS := \
	-DNETKIT_TARGET_MCU=1 \
	-DNETKIT_MCU_QUANT_ONLY=1 \
	-DNETKIT_KERNELS_OPTIMIZED_FOR_SPEED=1 \
	-DNETKIT_WEIGHTS_IN_RAM=$(NETKIT_WEIGHTS_IN_RAM) \
	$(if $(filter 1,$(NETKIT_CMSIS_DSP)),-DNETKIT_USE_CMSIS_DSP=1,) \
	$(if $(filter 1,$(NETKIT_CMSIS_NN)),-DNETKIT_USE_CMSIS_NN=1,) \
	-DNETKIT_QUANT_TRACE=0 \
	-DNETKIT_LOOP_UNROLL=0 \
	-DNETKIT_IM2COL=0 \
	-DNETKIT_REFERENCE_QUANT_LOOPS=$(NETKIT_REFERENCE_QUANT_LOOPS) \
	-DARM_MATH_CM4 \
	-DARM_MATH_LOOPUNROLL \
	-DARM_NN_ENABLE_F32=1

NETKIT_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/arena.cpp \
	$(NETKIT_ROOT)/src/tensor_factory.cpp \
	$(NETKIT_ROOT)/src/tensor_access.cpp \
	$(NETKIT_ROOT)/src/kernel_workspace.cpp \
	$(NETKIT_ROOT)/src/cmsis_buffer_size.cpp \
	$(NETKIT_ROOT)/src/reference_kernel.cpp \
	$(NETKIT_ROOT)/src/ops.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_backend.cpp \
	$(NETKIT_ROOT)/src/cmsis_nn_backend.cpp \
	$(NETKIT_ROOT)/src/quant_ops.cpp \
	$(NETKIT_ROOT)/src/quant_softmax_s8.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_util.cpp \
	$(NETKIT_ROOT)/src/cmsis_nn_quant_backend.cpp \
	$(NETKIT_ROOT)/src/mlp.cpp \
	$(NETKIT_ROOT)/src/nk_format.cpp \
	$(NETKIT_ROOT)/src/nk_loader.cpp

NETKIT_KERNEL_RUNTIME_SOURCES := \
	$(NETKIT_ROOT)/src/cmsis_nn_quant_backend.cpp \
	$(NETKIT_ROOT)/src/quant_softmax_s8.cpp \
	$(NETKIT_ROOT)/src/quant_ops.cpp \
	$(NETKIT_ROOT)/src/mlp.cpp

NETKIT_CORE_RUNTIME_SOURCES := $(filter-out $(NETKIT_KERNEL_RUNTIME_SOURCES),$(NETKIT_RUNTIME_SOURCES))

include $(NETKIT_ROOT)/third_party/cmsis_nn.mk

ifeq ($(NETKIT_REFERENCE_QUANT_LOOPS),1)
CMSIS_NN_SOURCES :=
else
CMSIS_NN_SOURCES := \
	$(CMSIS_NN_S8_SOURCES) \
	$(wildcard $(CMSIS_NN_DIR)/Source/FullyConnectedFunctions/arm_*_s8.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/ActivationFunctions/arm_*_s8.c) \
	$(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_softmax_s8.c \
	$(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c \
	$(wildcard $(CMSIS_NN_DIR)/Source/FullyConnectedFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/ActivationFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/BasicMathFunctions/arm_*f32*.c) \
	$(wildcard $(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_*f32*.c) \
	$(CMSIS_NN_DIR)/Source/NNSupportFunctions/arm_nntables_flt.c
endif

CMSIS_DSP_SOURCES := \
	$(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_q7.c \
	$(CMSIS_DSP_DIR)/Source/SupportFunctions/arm_copy_f32.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_q7.c \
	$(CMSIS_DSP_DIR)/Source/StatisticsFunctions/arm_max_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_q7.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_dot_prod_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_add_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_scale_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_offset_f32.c \
	$(CMSIS_DSP_DIR)/Source/BasicMathFunctions/arm_clip_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_init_f32.c \
	$(CMSIS_DSP_DIR)/Source/MatrixFunctions/arm_mat_vec_mult_f32.c

CFLAGS := $(MCU_CORE_CFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
CXXFLAGS := $(MCU_CORE_CXXFLAGS) $(MCU_DEBUG_FLAGS) $(INCLUDES) $(NETKIT_CPPFLAGS)
ASFLAGS := $(MCU_ASFLAGS) $(MCU_DEBUG_FLAGS)
LDFLAGS := $(MCU_LDFLAGS) $(NETKIT_CPPFLAGS) -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

LIBNETKIT := $(LIB_DIR)/libnetkit_mlp_int8.a

.PHONY: all clean flash monitor run export-embed check-deps size

all: check-deps export-embed $(ELF) size

check-deps:
	$(MCU_ASSERT_GCC_TOOLCHAIN)
	@test -f $(CMSIS_DSP_DIR)/Include/arm_math.h || \
		(echo "Run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(CMSIS_NN_DIR)/Include/arm_nnfunctions.h || \
		(echo "Run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/models/mnist_mlp_int8.nk || \
		(echo "Missing models/mnist_mlp_int8.nk — run: make -C $(NETKIT_ROOT) export-mnist-mlp-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_mlp_int8_test_images.cc || \
		(echo "Missing int8 test images — run: make -C $(NETKIT_ROOT) export-mnist-mlp-int8" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NETKIT_ROOT)/models/mnist_mlp_int8.nk
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python python3 -m netkit aot models/mnist_mlp_int8.nk \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		$(if $(filter 1,$(NETKIT_WEIGHTS_IN_RAM)),--weights-in-ram,--no-weights-in-ram) \
		--no-lower --omit-final-softmax
	@touch $(GEN_DIR)/.embed_stamp

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
	$(BUILD_DIR)/mnist_mlp_int8_test_images.o \
	$(BUILD_DIR)/mnist_mlp_int8_aot.o

$(BUILD_DIR)/startup.o: $(BOARD_COMMON)/startup/startup_stm32f446xx.S
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

$(BUILD_DIR)/mnist_mlp_int8_test_images.o: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_mlp_int8_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_mlp_int8_aot.o: $(GEN_DIR)/mnist_mlp_int8_aot.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ELF): $(OBJS) $(LIBNETKIT)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIBNETKIT) -lm -lc -lgcc -lstdc++ -lsupc++ -o $@
	$(OBJCOPY) -O binary $@ $(BIN)

size: $(ELF)
	$(SIZE) --format=berkeley $(ELF)

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(GEN_DIR)/.embed_stamp

flash: $(ELF)
	./scripts/flash.sh $(ELF)

monitor:
	./scripts/monitor.sh

run: flash monitor
