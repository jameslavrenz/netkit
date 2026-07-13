# netkit Makefile
#
# Primary build system (GNU Make). See docs/TESTING.md and docs/BUILD_TARGETS.md.
#
# Build target (NETKIT_TARGET) — ISA-qualified firmware profiles:
#   cpu       (default) — desktop: CLI, regression; arena defaults to heap
#   mcu_arm             — Arm MCU lean runtime; CMSIS-NN defaults (int8 production)
#   mpu_arm             — Arm MPU lean runtime; XNNPACK defaults
#   mcu_risc            — RISC-V MCU lean runtime (generic kernels; CMSIS-NN + XNNPACK forbidden)
#   mpu_risc            — RISC-V MPU lean runtime; XNNPACK on; CMSIS-NN forbidden
#
# Legacy NETKIT_TARGET=mcu|mpu is rejected — use mcu_arm / mpu_arm.
#
# Arena overrides:
#   NETKIT_GLOBAL_ARENA=1  — CPU only: use static/global arena instead of heap default
# NETKIT_HEAP_ARENA=1    — MPU class only: compile heap arena helpers (off by default).
#                          Forbidden on MCU (static/global arena only; no malloc/new).
#   NETKIT_ARENA_CAPACITY=<bytes> — override NK_ARENA_DEFAULT_CAPACITY (MCU default 64 KiB;
#                                   CPU/MPU default 64 MiB). Alternate: NETKIT_ARENA_KB=<KiB>.
#
# File-load mmap (POSIX macOS/Linux; Win32 on Windows):
#   NETKIT_MMAP=1          — enable mmap for LoadMLP/LoadCNN from path
#                            (default: cpu + any MPU = 1; MCU = forbidden)
#   NETKIT_MMAP=0          — fread into arena instead (use on RTOS/bare-metal MPU)
#
# Optional backends (profile defaults below; override on command line):
#   NETKIT_CMSIS_NN=1      — Arm MCU only (NETKIT_TARGET=mcu_arm + NETKIT_ARCH=CM4|M33|...)
#   NETKIT_XNNPACK=1       — Google XNNPACK (cpu + any MPU; forbidden on MCU; ./tools/fetch_xnnpack.sh)
#   CMSIS-DSP is not used (no NETKIT_CMSIS_DSP backend).
#
# Profile defaults:
#   cpu      — XNNPACK on (any host ISA), CMSIS-NN off
#   mcu_arm  — CMSIS-NN on, XNNPACK forbidden (float32 uses reference kernels)
#   mpu_arm  — XNNPACK on, CMSIS-NN off
#   mcu_risc — generic kernels only (CMSIS-NN + XNNPACK forbidden)
#   mpu_risc — XNNPACK on; CMSIS-NN forbidden
#
# Optional reference-kernel loop unroll (netkit code only; not CMSIS):
# NETKIT_IM2COL=0|1|2 — Conv2D strategy (float + int8 QuantOps): 0 direct (default), 1 partial, 2 full im2col+GEMM.
#   NETKIT_LOOP_UNROLL=1    — EXPERIMENTAL: 4× unroll in reference kernels (default 0).
#
# Target architecture (empty = desktop CPU; sets ARM_MATH_* flags for CMSIS):
#   NETKIT_ARCH=CM4|CM7|M33|M55|NEON|...  — see third_party/netkit_arch.mk
#
# Examples:
#   make                                    # desktop (cpu, heap arena)
#   make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all
#   make NETKIT_TARGET=mcu_arm NETKIT_ARCH=CM4 lib
#   make NETKIT_TARGET=mpu_arm lib

NETKIT_TARGET ?= cpu
NETKIT_GLOBAL_ARENA ?= 0
NETKIT_HEAP_ARENA ?= 0
# Optional default-arena override (bytes). Prefer NETKIT_ARENA_CAPACITY; NETKIT_ARENA_KB is KiB.
# NETKIT_ARENA_CAPACITY ?=
# NETKIT_ARENA_KB ?=
# mmap file load: default on for cpu + any MPU; forbidden on MCU (override with NETKIT_MMAP=0 on MPU/cpu).
ifneq ($(filter $(NETKIT_TARGET),mcu_arm mcu_risc),)
  NETKIT_MMAP ?= 0
else
  NETKIT_MMAP ?= 1
endif
# MCU: never allow mmap even if the caller forces NETKIT_MMAP=1.
ifneq ($(filter $(NETKIT_TARGET),mcu_arm mcu_risc),)
  ifeq ($(NETKIT_MMAP),1)
    $(warning NETKIT_MMAP=1 forced off on NETKIT_TARGET=$(NETKIT_TARGET) — mmap is forbidden on MCU)
  endif
  override NETKIT_MMAP := 0
endif
ifeq ($(NETKIT_TARGET),cpu)
  NETKIT_CMSIS_NN ?= 0
  NETKIT_XNNPACK ?= 1
else ifeq ($(NETKIT_TARGET),mcu_arm)
  NETKIT_CMSIS_NN ?= 1
  NETKIT_XNNPACK ?= 0
else ifeq ($(NETKIT_TARGET),mpu_arm)
  NETKIT_CMSIS_NN ?= 0
  NETKIT_XNNPACK ?= 1
else ifeq ($(NETKIT_TARGET),mcu_risc)
  NETKIT_CMSIS_NN ?= 0
  NETKIT_XNNPACK ?= 0
else ifeq ($(NETKIT_TARGET),mpu_risc)
  NETKIT_CMSIS_NN ?= 0
  NETKIT_XNNPACK ?= 1
else ifeq ($(NETKIT_TARGET),mcu)
  $(error NETKIT_TARGET=mcu is removed — use NETKIT_TARGET=mcu_arm)
else ifeq ($(NETKIT_TARGET),mpu)
  $(error NETKIT_TARGET=mpu is removed — use NETKIT_TARGET=mpu_arm)
else
  NETKIT_CMSIS_NN ?= 0
  NETKIT_XNNPACK ?= 0
endif
# In CI, cross-target host compile-checks keep reference kernels (no ARM toolchain /
# CMSIS-Core on the runner). XNNPACK forced off in CI.
ifeq ($(GITHUB_ACTIONS),true)
  ifneq ($(NETKIT_TARGET),cpu)
    override NETKIT_CMSIS_NN := 0
  endif
  override NETKIT_XNNPACK := 0
endif
# Default direct Conv2D loops (0) on all targets; override at build time if needed.
NETKIT_IM2COL ?= 0
NETKIT_LOOP_UNROLL ?= 0
NETKIT_ARCH ?=

include third_party/netkit_arch.mk

# Host embedded smoke (desktop): portable CMSIS-NN / ARM_MATH path without CMSIS-Core headers.
ifeq ($(NETKIT_HOST_SMOKE),1)
  NETKIT_ARCH_CFLAGS += -D__GNUC_PYTHON__
endif

CC ?= clang
CXX ?= clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

RUNTIME_SOURCES = src/arena.cpp src/nk_mmap.cpp src/tensor_factory.cpp src/tensor_access.cpp src/reference_kernel.cpp src/kernel_workspace.cpp src/cmsis_buffer_size.cpp src/ops.cpp \
                    src/conv2d.cpp src/depthwise_conv2d.cpp src/conv2d_layout.cpp src/conv_dispatch.cpp src/conv1x1_kernel.cpp src/conv_depthwise_kernel.cpp \
                    src/conv_direct_kernel.cpp src/im2col_partial.cpp src/im2col_full.cpp src/im2col_quant.cpp \
                    src/convnextv2_block.cpp src/mobilenetv4_uib.cpp src/resnet_basic_block.cpp src/yolox_decoupled_head.cpp src/yolox_pafpn.cpp src/mlp.cpp src/quant_ops.cpp src/quant_softmax_s8.cpp src/netkit_util.cpp src/quant_trace.cpp src/cmsis_nn_quant_backend.cpp src/cmsis_quant_plan.cpp src/cnn_quant.cpp src/cnn.cpp \
                    src/layer_ops/nk_op_conv2d.cpp src/layer_ops/nk_op_depthwise_conv2d.cpp \
                    src/layer_ops/nk_op_convnextv2_block.cpp src/layer_ops/nk_op_mobilenetv4_uib.cpp src/layer_ops/nk_op_yolox_decoupled_head.cpp src/layer_ops/nk_op_feature_tap.cpp src/layer_ops/nk_op_yolox_pafpn.cpp src/layer_ops/nk_op_resnet_basic_block.cpp src/layer_ops/nk_op_layernorm2d.cpp \
                    src/layer_ops/nk_op_max_pool2d.cpp \
                    src/layer_ops/nk_op_avg_pool2d.cpp src/layer_ops/nk_op_batch_norm2d.cpp \
                    src/layer_ops/nk_op_flatten.cpp src/layer_ops/nk_op_dense.cpp \
                    src/ops_resolver.cpp src/ops_resolver_default.cpp \
                    src/nk_format.cpp src/nk_loader.cpp src/netkit_api.cpp

TRIM_LAYER_OP_SOURCES = src/layer_ops/nk_op_conv2d.cpp src/layer_ops/nk_op_max_pool2d.cpp \
                        src/layer_ops/nk_op_flatten.cpp src/layer_ops/nk_op_dense.cpp

TARGET_CPPFLAGS = $(NETKIT_ARCH_CFLAGS)

CMSIS_NN_OBJECTS =
NETKIT_CMSIS_NN_EFFECTIVE := 0
CMSIS_SOFTMAX_OBJECTS =
CMSIS_NN_DIR ?= third_party/CMSIS-NN
# RISC targets: CMSIS-NN is never allowed (generic kernels only for those).
ifneq ($(filter $(NETKIT_TARGET),mcu_risc mpu_risc),)
  ifeq ($(NETKIT_CMSIS_NN),1)
    $(warning NETKIT_CMSIS_NN=1 forced off on NETKIT_TARGET=$(NETKIT_TARGET) — CMSIS-NN is forbidden on RISC)
  endif
  override NETKIT_CMSIS_NN := 0
endif
ifneq ($(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c),)
  CMSIS_SOFTMAX_CFLAGS = -std=c11 -O2 -I$(CMSIS_NN_DIR)/Include
  CMSIS_SOFTMAX_SOURCES = \
    $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c
  CMSIS_SOFTMAX_OBJECTS = $(CMSIS_SOFTMAX_SOURCES:$(CMSIS_NN_DIR)/%.c=build/cmsis_softmax/%.o)
  TARGET_CPPFLAGS += -DNETKIT_USE_CMSIS_SOFTMAX_S8=1 -I$(CMSIS_NN_DIR)/Include
endif
ifeq ($(NETKIT_CMSIS_NN),1)
  ifeq ($(NETKIT_TARGET),mcu_arm)
    ifeq ($(NETKIT_ARCH_IS_M_PROFILE),1)
      NETKIT_CMSIS_NN_EFFECTIVE := 1
    else
      $(warning NETKIT_CMSIS_NN=1 ignored — set NETKIT_ARCH=CM4|M33|... (Cortex-M); using reference kernels)
    endif
  else
    $(warning NETKIT_CMSIS_NN=1 ignored on NETKIT_TARGET=$(NETKIT_TARGET) — CMSIS-NN is mcu_arm + Cortex-M only)
  endif
endif
ifeq ($(NETKIT_CMSIS_NN_EFFECTIVE),1)
  CMSIS_NN_DIR ?= third_party/CMSIS-NN
  ifeq ($(wildcard $(CMSIS_NN_DIR)/Include/arm_nnfunctions.h),)
    $(error NETKIT_CMSIS_NN=1 requires CMSIS-NN at $(CMSIS_NN_DIR) — run ./tools/fetch_cmsis_nn.sh)
  endif
  include third_party/cmsis_nn.mk
  TARGET_CPPFLAGS += -DNETKIT_USE_CMSIS_NN=1 -DARM_NN_ENABLE_F32=1 -I$(CMSIS_NN_DIR)/Include
  RUNTIME_SOURCES += src/cmsis_nn_backend.cpp
endif


# XNNPACK: default LayerFast on cpu + any MPU. Always forced off on MCU.
XNNPACK_DIR ?= third_party/XNNPACK
XNNPACK_LIB_DIR ?= $(XNNPACK_DIR)/netkit_lib
XNNPACK_LDFLAGS :=
NETKIT_XNNPACK_EFFECTIVE := 0
ifneq ($(filter $(NETKIT_TARGET),mcu_arm mcu_risc),)
  ifeq ($(NETKIT_XNNPACK),1)
    $(warning NETKIT_XNNPACK=1 forced off on NETKIT_TARGET=$(NETKIT_TARGET) — XNNPACK is forbidden on MCU)
  endif
  override NETKIT_XNNPACK := 0
endif
ifeq ($(NETKIT_XNNPACK),1)
  ifeq ($(wildcard $(XNNPACK_DIR)/include/xnnpack.h),)
    $(warning NETKIT_XNNPACK=1 but XNNPACK headers missing — run ./tools/fetch_xnnpack.sh; using reference kernels)
  else ifeq ($(wildcard $(XNNPACK_LIB_DIR)/libXNNPACK.a),)
    $(warning NETKIT_XNNPACK=1 but libXNNPACK.a missing — run ./tools/fetch_xnnpack.sh; using reference kernels)
  else
    NETKIT_XNNPACK_EFFECTIVE := 1
  endif
endif
ifeq ($(NETKIT_XNNPACK_EFFECTIVE),1)
  TARGET_CPPFLAGS += -DNETKIT_USE_XNNPACK=1 -I$(XNNPACK_DIR)/include -I$(XNNPACK_DIR)/netkit_include
  RUNTIME_SOURCES += src/xnnpack_backend.cpp src/xnnpack_float_backend.cpp
  # macOS needs force_load so microkernel registrars are not stripped.
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
# Int8 qs8 stubs always compile (no-op when XNNPACK off); real path when enabled.
RUNTIME_SOURCES += src/xnnpack_quant_backend.cpp

DESKTOP_SOURCES = src/nk_regression.cpp src/cli.cpp src/test.cpp

ifeq ($(NETKIT_TARGET),cpu)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_CPU=1
  CORE_SOURCES = $(RUNTIME_SOURCES) $(DESKTOP_SOURCES)
  BUILD_CLI = 1
  BUILD_C_TESTS = 1
  ifeq ($(NETKIT_GLOBAL_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_GLOBAL_ARENA=1
  endif
else ifeq ($(NETKIT_TARGET),mcu_arm)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MCU_ARM=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    $(error NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena buffer)
  endif
else ifeq ($(NETKIT_TARGET),mpu_arm)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MPU_ARM=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_HEAP_ARENA=1
  endif
else ifeq ($(NETKIT_TARGET),mcu_risc)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MCU_RISC=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    $(error NETKIT_HEAP_ARENA is forbidden on MCU — use a static/global arena buffer)
  endif
else ifeq ($(NETKIT_TARGET),mpu_risc)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MPU_RISC=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_HEAP_ARENA=1
  endif
else
  $(error NETKIT_TARGET must be cpu, mcu_arm, mpu_arm, mcu_risc, or mpu_risc (got '$(NETKIT_TARGET)'))
endif

TARGET_CPPFLAGS += -DNETKIT_IM2COL=$(NETKIT_IM2COL)
TARGET_CPPFLAGS += -DNETKIT_LOOP_UNROLL=$(NETKIT_LOOP_UNROLL)
TARGET_CPPFLAGS += -DNETKIT_USE_MMAP=$(NETKIT_MMAP)
ifdef NETKIT_ARENA_CAPACITY
  TARGET_CPPFLAGS += -DNK_ARENA_DEFAULT_CAPACITY=$(NETKIT_ARENA_CAPACITY)
else ifdef NETKIT_ARENA_KB
  TARGET_CPPFLAGS += -DNK_ARENA_DEFAULT_CAPACITY=$(shell echo $$(($(NETKIT_ARENA_KB) * 1024)))
endif

CFLAGS += $(TARGET_CPPFLAGS)
CXXFLAGS += $(TARGET_CPPFLAGS)

# Always -O2 for CPU/MPU builds and tests (never -Os). Matches TFLM kernel speed.
CFLAGS += -O2
CXXFLAGS += -O2

CLI_SOURCES = src/main.cpp

CORE_OBJECTS = $(CORE_SOURCES:.cpp=.o)
CLI_OBJECTS = $(CLI_SOURCES:.cpp=.o)

EXAMPLE_C = examples/infer_c
EXAMPLE_C_SRC = examples/infer_c.c
EXAMPLE_C_OBJ = examples/infer_c.o

EXAMPLE_CPP = examples/infer_cpp
EXAMPLE_CPP_SRC = examples/infer_cpp.cpp
EXAMPLE_CPP_OBJ = examples/infer_cpp.o

TEST_C = tests/test_c_api
TEST_C_SRC = tests/test_c_api.c
TEST_C_OBJ = tests/test_c_api.o

EMBEDDED_SMOKE = tests/embedded_smoke
EMBEDDED_SMOKE_SRC = tests/embedded_smoke.c
EMBEDDED_SMOKE_OBJ = tests/embedded_smoke.o

NK_INFER = tools/nk_infer
NK_INFER_SRC = tools/nk_infer.c
NK_INFER_OBJ = tools/nk_infer.o

TRIM_LIB = libnetkit_trim.a
TRIM_RUNTIME_SOURCES = src/arena.cpp src/nk_mmap.cpp src/tensor_factory.cpp src/tensor_access.cpp src/reference_kernel.cpp src/kernel_workspace.cpp src/cmsis_buffer_size.cpp src/ops.cpp \
                       src/conv2d.cpp src/conv2d_layout.cpp src/conv_dispatch.cpp src/conv1x1_kernel.cpp src/conv_depthwise_kernel.cpp \
                       src/conv_direct_kernel.cpp src/im2col_partial.cpp src/im2col_full.cpp src/im2col_quant.cpp src/mlp.cpp src/cnn.cpp $(TRIM_LAYER_OP_SOURCES) src/ops_resolver.cpp \
                       src/nk_format.cpp src/nk_loader.cpp src/netkit_api.cpp
TRIM_CORE_OBJECTS = $(TRIM_RUNTIME_SOURCES:.cpp=.o)

# Rebuild objects when target/backends change — avoids mixing CPU and MCU .o files in libnetkit.a.
NETKIT_BUILD_STAMP = .netkit_build_stamp
NETKIT_BUILD_ID = target=$(NETKIT_TARGET),global_arena=$(NETKIT_GLOBAL_ARENA),heap_arena=$(NETKIT_HEAP_ARENA),arena_cap=$(NETKIT_ARENA_CAPACITY)$(NETKIT_ARENA_KB),mmap=$(NETKIT_MMAP),cmsis_nn=$(NETKIT_CMSIS_NN_EFFECTIVE),xnnpack=$(NETKIT_XNNPACK_EFFECTIVE),im2col=$(NETKIT_IM2COL),loop_unroll=$(NETKIT_LOOP_UNROLL),arch=$(NETKIT_ARCH),host_smoke=$(NETKIT_HOST_SMOKE)

NETKIT_STALE_BINARIES = $(TARGET) $(LIB) $(TRIM_LIB) $(EXAMPLE_C) $(EXAMPLE_CPP) $(TEST_C) $(EMBEDDED_SMOKE) $(NK_INFER)

.PHONY: netkit-config-sync
netkit-config-sync:
	@printf '%s\n' '$(NETKIT_BUILD_ID)' > $(NETKIT_BUILD_STAMP).tmp
	@if ! [ -f $(NETKIT_BUILD_STAMP) ] || ! cmp -s $(NETKIT_BUILD_STAMP).tmp $(NETKIT_BUILD_STAMP); then \
	  echo "netkit build config changed — rebuilding"; \
	  mv $(NETKIT_BUILD_STAMP).tmp $(NETKIT_BUILD_STAMP); \
	  rm -f $(NETKIT_STALE_BINARIES); \
	else \
	  rm -f $(NETKIT_BUILD_STAMP).tmp; \
	fi

# Real stamp target so parallel `make -j` can create the file after `clean`
# (%.o depends on it; phony netkit-config-sync alone is not enough).
$(NETKIT_BUILD_STAMP):
	@$(MAKE) --no-print-directory netkit-config-sync

.PHONY: all lib clean rebuild test test-full test-cpp test-c test-python test-python-full run example-c example-cpp examples \
        export-mnist export-mnist-int8 export-mnist-cnn export-mnist-all export-op-matrix \
        export-nk build-all embed-tests cmsis-nn-init cmsis-init \
        cpu cpu-global mcu-arm mpu-arm mpu-arm-heap mcu-risc mpu-risc \
        embedded-smoke test-embedded-smoke test-embedded-smoke-matrix trim-lib check-trim-lib

ifeq ($(BUILD_CLI),1)
all: netkit-config-sync $(TARGET)
build-all: netkit-config-sync all examples $(TEST_C)
else
all: netkit-config-sync $(LIB)
build-all: netkit-config-sync $(LIB) examples embedded-smoke
endif

.DEFAULT_GOAL := all

$(LIB): $(CORE_OBJECTS) $(CMSIS_NN_OBJECTS) $(CMSIS_SOFTMAX_OBJECTS)
	ar rcs $@ $^

build/cmsis_softmax/%.o: $(CMSIS_NN_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CMSIS_SOFTMAX_CFLAGS) -c $< -o $@

lib: netkit-config-sync $(LIB)

$(TRIM_LIB): $(TRIM_CORE_OBJECTS)
	ar rcs $@ $^

trim-lib: $(TRIM_LIB)

check-trim-lib:
	$(MAKE) NETKIT_TARGET=cpu lib trim-lib
	chmod +x tools/check_trim_lib.sh && ./tools/check_trim_lib.sh

ifeq ($(BUILD_CLI),1)
$(TARGET): netkit-config-sync $(LIB) $(CLI_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB) $(XNNPACK_LDFLAGS)
endif

$(EXAMPLE_C): netkit-config-sync $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB) $(XNNPACK_LDFLAGS)

$(EXAMPLE_CPP): netkit-config-sync $(LIB) $(EXAMPLE_CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_CPP_OBJ) $(LIB) $(XNNPACK_LDFLAGS)

ifeq ($(BUILD_C_TESTS),1)
$(TEST_C): netkit-config-sync $(LIB) $(TEST_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_C_OBJ) $(LIB) $(XNNPACK_LDFLAGS)
endif

$(EMBEDDED_SMOKE): netkit-config-sync $(LIB) $(EMBEDDED_SMOKE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EMBEDDED_SMOKE_OBJ) $(LIB) $(XNNPACK_LDFLAGS)

ifeq ($(BUILD_CLI),1)
$(NK_INFER): netkit-config-sync $(LIB) $(NK_INFER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(NK_INFER_OBJ) $(LIB) $(XNNPACK_LDFLAGS)
endif

%.o: %.cpp $(NETKIT_BUILD_STAMP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_CPP_OBJ): $(EXAMPLE_CPP_SRC) $(NETKIT_BUILD_STAMP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(NK_INFER_OBJ): $(NK_INFER_SRC) include/netkit.h $(NETKIT_BUILD_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

$(EXAMPLE_C_OBJ): $(EXAMPLE_C_SRC) include/netkit.h $(NETKIT_BUILD_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(BUILD_C_TESTS),1)
$(TEST_C_OBJ): $(TEST_C_SRC) include/netkit.h $(NETKIT_BUILD_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(EMBEDDED_SMOKE_OBJ): $(EMBEDDED_SMOKE_SRC) include/netkit.h $(NETKIT_BUILD_STAMP)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(NETKIT_BUILD_STAMP)
	rm -f $(CORE_OBJECTS) $(TRIM_CORE_OBJECTS) $(CLI_OBJECTS) $(EXAMPLE_C_OBJ) $(EXAMPLE_CPP_OBJ) $(TEST_C_OBJ) $(EMBEDDED_SMOKE_OBJ) $(NK_INFER_OBJ) \
	      $(TARGET) $(LIB) $(TRIM_LIB) $(EXAMPLE_C) $(EXAMPLE_CPP) $(TEST_C) $(EMBEDDED_SMOKE) $(NK_INFER) examples/trim_firmware examples/trim_firmware.o
	rm -f src/*.o src/layer_ops/*.o examples/*.o tests/*.o tools/*.o
	rm -rf build/cmsis_nn build/cmsis_softmax

rebuild: clean all

ifeq ($(BUILD_CLI),1)
test-cpp: $(TARGET)
	./$(TARGET) test
else
test-cpp:
	@echo "test-cpp requires NETKIT_TARGET=cpu (got $(NETKIT_TARGET))" >&2
	@exit 1
endif

ifeq ($(BUILD_C_TESTS),1)
test-c: $(TEST_C)
	./$(TEST_C)
else
test-c:
	@echo "test-c requires NETKIT_TARGET=cpu (got $(NETKIT_TARGET))" >&2
	@exit 1
endif

test: test-cpp test-c test-python check-trim-lib

test-full: test-cpp test-c test-python-full check-trim-lib

# Backward-compatible alias
test-fast: test

test-python: lib $(TARGET) $(NK_INFER)
	NETKIT_FAST_TESTS=1 PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py'

test-python-full: lib $(TARGET) $(NK_INFER)
	PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py'

run: test

example-c: $(EXAMPLE_C)

example-cpp: $(EXAMPLE_CPP)

examples: example-cpp example-c

embedded-smoke: $(EMBEDDED_SMOKE)

test-embedded-smoke: embedded-smoke
	./$(EMBEDDED_SMOKE)

test-embedded-smoke-matrix:
	./tools/run_embedded_smoke.sh

cpu:
	$(MAKE) NETKIT_TARGET=cpu all

cpu-global:
	$(MAKE) NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all

mcu-arm:
	$(MAKE) NETKIT_TARGET=mcu_arm lib

mpu-arm:
	$(MAKE) NETKIT_TARGET=mpu_arm lib

mpu-arm-heap:
	$(MAKE) NETKIT_TARGET=mpu_arm NETKIT_HEAP_ARENA=1 lib

mcu-risc:
	$(MAKE) NETKIT_TARGET=mcu_risc lib

mpu-risc:
	$(MAKE) NETKIT_TARGET=mpu_risc lib

cmsis-nn-init:
	./tools/fetch_cmsis_nn.sh

cmsis-core-init:
	./tools/fetch_cmsis_core.sh

xnnpack-init:
	./tools/fetch_xnnpack.sh

cmsis-init: cmsis-nn-init cmsis-core-init

.PHONY: xnnpack-init

export-mnist:
	PYTHONPATH=python python3 tools/export_mnist_mlp.py

export-mnist-int8:
	PYTHONPATH=python python3 tools/export_mnist_mlp_int8.py

export-mnist-cnn-int8:
	PYTHONPATH=python python3 tools/export_mnist_cnn_int8.py --from-nk models/mnist_cnn.nk --fast
	python3 benchmark/tflm/tools/export_int8_test_images.py --variant cnn

export-mnist-mlp-int8:
	/tmp/netkit-tf-venv/bin/python benchmark/tflm/tools/export_mnist_mlp_int8_tflite.py
	PYTHONPATH=python python3 tools/export_mnist_mlp_int8.py --from-nk models/mnist_mlp.nk --align-tflite benchmark/tflm/generated/mnist_mlp_int8.tflite
	/tmp/netkit-tf-venv/bin/python benchmark/tflm/tools/export_int8_test_images.py --variant mlp

export-mnist-cnn-int8-retrain:
	PYTHONPATH=python python3 tools/export_mnist_cnn_int8.py --retrain

flash-mnist-cnn-int8:
	cd boards/nucleo-f446re-cnn-int8 && chmod +x scripts/*.sh && ./scripts/deploy.sh all

export-mnist-cnn:
	PYTHONPATH=python python3 tools/export_mnist_cnn.py

export-mnist-all: export-mnist export-mnist-cnn
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-op-matrix:
	PYTHONPATH=python python3 tools/write_op_matrix_models.py
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-import-parity:
	PYTHONPATH=python python3 tools/write_import_parity_models.py

export-onnx-test:
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-nk:
	PYTHONPATH=python python3 -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk
	PYTHONPATH=python python3 -m netkit convert models/mlp_hand.onnx -o models/mlp_hand.nk
	PYTHONPATH=python python3 -m netkit convert models/test_cnn.onnx -o models/test_cnn.nk
	PYTHONPATH=python python3 -m netkit convert models/cnn_4x4_single.onnx -o models/cnn_4x4_single.nk
	PYTHONPATH=python python3 -m netkit convert models/cnn_hand.onnx -o models/cnn_hand.nk
	PYTHONPATH=python python3 tools/embed_nk_tests.py

embed-tests:
	PYTHONPATH=python python3 tools/embed_nk_tests.py
