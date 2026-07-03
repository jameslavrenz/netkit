# Bare-metal Cortex-M4F hand-model benchmark for Arm FVP (reference vs CMSIS-NN + CMSIS-DSP).
#
# Usage (from repo root):
#   make -C benchmarks/fvp VARIANT=ref MODEL=mlp all
#   make -C benchmarks/fvp VARIANT=cmsis MODEL=cnn all
#
# Or: make bench-hand-fvp

ROOT := ../..
VARIANT ?= ref
MODEL ?= mlp
BUILD := build/$(VARIANT)-$(MODEL)
OUT := hand_fvp_bench_$(VARIANT)_$(MODEL).elf

ifeq ($(MODEL),mlp)
  MODEL_FLAG := -DNETKIT_FVP_MODEL_MLP=1
else ifeq ($(MODEL),cnn)
  MODEL_FLAG := -DNETKIT_FVP_MODEL_CNN=1
else
  $(error MODEL must be mlp or cnn)
endif

export CC := arm-none-eabi-gcc
export CXX := arm-none-eabi-g++
export AR := arm-none-eabi-ar

NETKIT_TARGET := mcu
NETKIT_ARCH := CM4
CPU_FLAGS := -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

include $(ROOT)/third_party/netkit_arch.mk

COMMON_CFLAGS := -g -O2 -Wall -Wextra -I$(ROOT)/include $(CPU_FLAGS) $(NETKIT_ARCH_CFLAGS) -DNETKIT_TARGET_MCU=1 -DNETKIT_DISABLE_IOSTREAM=1 -ffunction-sections -fdata-sections
COMMON_CXXFLAGS := $(COMMON_CFLAGS) -std=c++26 -fno-exceptions -fno-rtti
BENCH_CFLAGS := $(COMMON_CFLAGS) -std=c23 $(MODEL_FLAG) -DNETKIT_FVP_CYCLE_TARGET=100

RUNTIME_SOURCES := \
	arena.cpp tensor_factory.cpp tensor_access.cpp reference_kernel.cpp ops.cpp \
	conv2d.cpp mlp.cpp cnn.cpp \
	layer_ops/nk_op_conv2d.cpp layer_ops/nk_op_max_pool2d.cpp layer_ops/nk_op_avg_pool2d.cpp \
	layer_ops/nk_op_batch_norm2d.cpp layer_ops/nk_op_flatten.cpp layer_ops/nk_op_dense.cpp \
	ops_resolver.cpp ops_resolver_default.cpp \
	nk_format.cpp nk_loader.cpp netkit_api.cpp

RUNTIME_OBJS := $(patsubst %.cpp,$(BUILD)/src/%.o,$(RUNTIME_SOURCES))
BENCH_OBJS := $(BUILD)/hand_fvp_bench.o $(BUILD)/hand_fvp_embedded.o $(BUILD)/startup_cm4.o

ifeq ($(VARIANT),cmsis)
  BACKEND := cmsis-nn+dsp
  NETKIT_CMSIS_NN := 1
  NETKIT_CMSIS_DSP := 1
  CMSIS_NN_DIR := $(ROOT)/third_party/CMSIS-NN
  CMSIS_DSP_DIR := $(ROOT)/third_party/CMSIS-DSP
  CMSISCORE_DIR := $(ROOT)/third_party/CMSIS-Core/Include
  BENCH_CFLAGS += -DNETKIT_BENCH_BACKEND=\"$(BACKEND)\"
  RUNTIME_CPPFLAGS := $(NETKIT_ARCH_CFLAGS) -DNETKIT_USE_CMSIS_NN=1 -DNETKIT_USE_CMSIS_DSP=1 \
	-DARM_NN_ENABLE_F32=1 -DNETKIT_TARGET_MCU=1 \
	-I$(CMSIS_NN_DIR)/Include -I$(CMSIS_DSP_DIR)/Include -I$(CMSIS_DSP_DIR)/PrivateInclude -I$(CMSISCORE_DIR)
  include $(ROOT)/third_party/cmsis_nn.mk
  include $(ROOT)/third_party/cmsis_dsp.mk
  RUNTIME_SOURCES += cmsis_nn_backend.cpp cmsis_dsp_backend.cpp
  RUNTIME_OBJS := $(patsubst %.cpp,$(BUILD)/src/%.o,$(RUNTIME_SOURCES))
  CMSIS_NN_OBJECTS := $(CMSIS_NN_SOURCES:$(CMSIS_NN_DIR)/%.c=$(BUILD)/cmsis_nn/%.o)
  CMSIS_DSP_OBJECTS := $(CMSIS_DSP_SOURCES:$(CMSIS_DSP_DIR)/%.c=$(BUILD)/cmsis_dsp/%.o)
  EXTRA_OBJS := $(CMSIS_NN_OBJECTS) $(CMSIS_DSP_OBJECTS)
else ifeq ($(VARIANT),ref)
  BACKEND := reference
  BENCH_CFLAGS += -DNETKIT_BENCH_BACKEND=\"$(BACKEND)\"
  RUNTIME_CPPFLAGS := $(NETKIT_ARCH_CFLAGS) -DNETKIT_TARGET_MCU=1
  EXTRA_OBJS :=
else
  $(error VARIANT must be ref or cmsis)
endif

RUNTIME_CXXFLAGS := $(COMMON_CXXFLAGS) $(RUNTIME_CPPFLAGS)
LDFLAGS := $(CPU_FLAGS) -Tlinker_cm4.ld --specs=rdimon.specs -lc -lrdimon -Wl,--gc-sections

.PHONY: all clean all-models

all: $(OUT)

all-models:
	$(MAKE) VARIANT=ref MODEL=mlp all
	$(MAKE) VARIANT=ref MODEL=cnn all
	$(MAKE) VARIANT=cmsis MODEL=mlp all
	$(MAKE) VARIANT=cmsis MODEL=cnn all

$(OUT): $(RUNTIME_OBJS) $(BENCH_OBJS) $(EXTRA_OBJS) linker_cm4.ld
	$(CXX) $(LDFLAGS) -o $@ $(BENCH_OBJS) $(RUNTIME_OBJS) $(EXTRA_OBJS)

$(BUILD)/src/%.o: $(ROOT)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(RUNTIME_CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(BENCH_CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.S
	@mkdir -p $(dir $@)
	$(CC) $(COMMON_CFLAGS) -c $< -o $@

ifeq ($(VARIANT),cmsis)
$(BUILD)/cmsis_nn/%.o: $(CMSIS_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_NN_CFLAGS) $(CPU_FLAGS) -I$(CMSISCORE_DIR) -c $< -o $@

$(BUILD)/cmsis_dsp/%.o: $(CMSIS_DSP_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_DSP_CFLAGS) $(CPU_FLAGS) -I$(CMSISCORE_DIR) -c $< -o $@
endif

clean:
	rm -rf build/$(VARIANT)-$(MODEL) hand_fvp_bench_$(VARIANT)_$(MODEL).elf
