# NUCLEO-F446RE (STM32F446RET6) — MNIST MLP f32 benchmark firmware.
#
# Build profile:
#   NETKIT_TARGET=mcu, NETKIT_ARCH=CM4, NETKIT_CMSIS_DSP=1, NETKIT_WEIGHTS_IN_RAM=0
#
# Usage:
#   make cmsis-init          # once, from repo root
#   make                     # build firmware.elf
#   ./scripts/flash.sh       # program via ST-Link (USB)
#   ./scripts/monitor.sh     # read UART results (ST-Link VCP)

NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)

MCU_BOARD_COMMON := $(BOARD_ROOT)
include $(BOARD_ROOT)/mcu_gcc_toolchain.mk

MCU := cortex-m4
BUILD_DIR := build
MCU_FLAGS := -mcpu=$(MCU) -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
OPT_FLAGS := -Os -ffunction-sections -fdata-sections -flto
WARN_FLAGS := -Wall -Wextra
GEN_DIR := generated
LIB_DIR := $(BUILD_DIR)/lib

TARGET := mnist_mlp_nucleo_f446re
ELF := $(BUILD_DIR)/$(TARGET).elf
BIN := $(BUILD_DIR)/$(TARGET).bin

LINKER_SCRIPT := linker/STM32F446RETx_FLASH.ld

INCLUDES := \
	-I$(BOARD_ROOT)/include \
	-I$(NETKIT_ROOT)/include \
	-I$(GEN_DIR) \
	-I$(NETKIT_ROOT)/benchmark/tflm/generated \
	-I$(NETKIT_ROOT)/third_party/CMSIS-DSP/Include \
	-I$(NETKIT_ROOT)/third_party/CMSIS-DSP/PrivateInclude \
	-I$(NETKIT_ROOT)/third_party/CMSIS-Core/CMSIS/Core/Include

# Override: make NETKIT_WEIGHTS_IN_RAM=1  (MNIST needs ~489 KiB arena — exceeds 128 KiB SRAM on F446)
NETKIT_WEIGHTS_IN_RAM ?= 0

NETKIT_CPPFLAGS := \
	-DNETKIT_TARGET_MCU=1 \
	-DNETKIT_WEIGHTS_IN_RAM=$(NETKIT_WEIGHTS_IN_RAM) \
	-DNETKIT_USE_CMSIS_DSP=1 \
	-DNETKIT_LOOP_UNROLL=0 \
	-DNETKIT_IM2COL_FULL=0 \
	-DARM_MATH_CM4 \
	-DARM_MATH_LOOPUNROLL

NETKIT_MLP_SOURCES := \
	$(NETKIT_ROOT)/src/arena.cpp \
	$(NETKIT_ROOT)/src/tensor_factory.cpp \
	$(NETKIT_ROOT)/src/tensor_access.cpp \
	$(NETKIT_ROOT)/src/reference_kernel.cpp \
	$(NETKIT_ROOT)/src/kernel_workspace.cpp \
	$(NETKIT_ROOT)/src/cmsis_buffer_size.cpp \
	$(NETKIT_ROOT)/src/ops.cpp \
	$(NETKIT_ROOT)/src/cmsis_dsp_backend.cpp

CMSIS_DSP_SOURCES := \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_add_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_mult_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_scale_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_offset_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/BasicMathFunctions/arm_clip_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/StatisticsFunctions/arm_mean_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/StatisticsFunctions/arm_var_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/MatrixFunctions/arm_mat_init_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
	$(NETKIT_ROOT)/third_party/CMSIS-DSP/Source/MatrixFunctions/arm_mat_mult_f32.c

BOARD_C_SOURCES := \
	src/system_stm32f4xx.c \
	src/uart.c \
	src/dwt_time.c \
	src/syscalls.c

BOARD_CXX_SOURCES := \
	src/main.cpp \
	$(NETKIT_ROOT)/benchmark/tflm/generated/mnist_test_images.cc

BOARD_ASM_SOURCES := startup/startup_stm32f446xx.S

GEN_SOURCES := \
	$(GEN_DIR)/mnist_mlp_aot.cpp

CFLAGS := $(MCU_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) -std=c23 -g $(INCLUDES) $(NETKIT_CPPFLAGS)
CXXFLAGS := $(MCU_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS) -std=c++20 -g $(INCLUDES) $(NETKIT_CPPFLAGS) -fno-exceptions -fno-rtti
ASFLAGS := $(MCU_FLAGS) -g
LDFLAGS := $(MCU_FLAGS) -flto -Wl,--gc-sections -T$(LINKER_SCRIPT) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map

LIBNETKIT := $(LIB_DIR)/libnetkit_mnist_mlp.a

.PHONY: all clean flash monitor run export-embed check-deps size

all: check-deps export-embed $(ELF) size

check-deps:
	$(MCU_ASSERT_GCC_TOOLCHAIN)
	@test -f $(NETKIT_ROOT)/third_party/CMSIS-DSP/Include/arm_math.h || \
		(echo "Run: make -C $(NETKIT_ROOT) cmsis-init" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/models/mnist_mlp.nk || \
		(echo "Missing models/mnist_mlp.nk — run: make -C $(NETKIT_ROOT) export-mnist" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NETKIT_ROOT)/models/mnist_mlp.nk $(NETKIT_ROOT)/python/netkit/aot_lower.py
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python python3 -m netkit aot models/mnist_mlp.nk \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --optimize --arena-headroom 20 \
		$(if $(filter 1,$(NETKIT_WEIGHTS_IN_RAM)),--weights-in-ram,--no-weights-in-ram)
	@touch $(GEN_DIR)/.embed_stamp

export-aot: export-embed

$(LIBNETKIT): $(NETKIT_MLP_SOURCES) $(CMSIS_DSP_SOURCES)
	@mkdir -p $(LIB_DIR)
	@rm -f $(LIB_DIR)/*.o
	@set -e; for src in $(CMSIS_DSP_SOURCES); do \
		obj=$(LIB_DIR)/cmsis_$$(basename $$src .c).o; \
		$(CC) $(CFLAGS) -c $$src -o $$obj; \
	done
	@set -e; for src in $(NETKIT_MLP_SOURCES); do \
		obj=$(LIB_DIR)/netkit_$$(basename $$src .cpp).o; \
		$(CXX) $(CXXFLAGS) -c $$src -o $$obj; \
	done
	$(AR) rcs $@ $(LIB_DIR)/*.o

OBJS := \
	$(BUILD_DIR)/startup.o \
	$(BUILD_DIR)/system_stm32f4xx.o \
	$(BUILD_DIR)/uart.o \
	$(BUILD_DIR)/dwt_time.o \
	$(BUILD_DIR)/syscalls.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/mnist_test_images.o \
	$(BUILD_DIR)/mnist_mlp_aot.o

$(BUILD_DIR)/startup.o: startup/startup_stm32f446xx.S
	@mkdir -p $(BUILD_DIR)
	$(CC) $(ASFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: src/main.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_test_images.o: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_test_images.cc
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/mnist_mlp_aot.o: $(GEN_DIR)/mnist_mlp_aot.cpp
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
	rm -rf $(GEN_DIR)

flash: $(ELF)
	./scripts/flash.sh $(ELF)

monitor:
	./scripts/monitor.sh

run: flash monitor
