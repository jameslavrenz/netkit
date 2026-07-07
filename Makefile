# netkit Makefile
#
# Primary build system (GNU Make). See docs/TESTING.md and docs/BUILD_TARGETS.md.
#
# Build target (NETKIT_TARGET):
#   cpu (default) — desktop: CLI, regression; arena defaults to heap
#   mcu           — lean runtime only; arena defaults to caller-owned global/static buffer
#   mpu           — lean runtime only; same arena default as MCU
#
# Arena overrides:
#   NETKIT_GLOBAL_ARENA=1  — CPU only: use static/global arena instead of heap default
#   NETKIT_HEAP_ARENA=1    — MCU/MPU: compile heap arena helpers (off by default)
#
# Weight load policy (buffer / AOT path only; file load always copies to arena):
#   NETKIT_WEIGHTS_IN_RAM=1|0 — default 0 (coefs in flash/blob); set 1 to copy payload to SRAM
#
# Optional CMSIS backends (opt-in via NETKIT_CMSIS_*=1; profile defaults below):
#   NETKIT_CMSIS_NN=1      — Cortex-M MCU only (NETKIT_TARGET=mcu + NETKIT_ARCH=CM4|M33|...)
#   NETKIT_CMSIS_DSP=1     — CMSIS-DSP float32 vector/matrix ops
#
# Profile defaults (override on command line, e.g. make NETKIT_CMSIS_DSP=0):
#   cpu — CMSIS-DSP on, CMSIS-NN off
#   mcu — both on (requires NETKIT_ARCH=CM4|M33|... for NN)
#   mpu — CMSIS-DSP on, CMSIS-NN off
#
# Optional reference-kernel loop unroll (netkit code only; not CMSIS):
#   NETKIT_IM2COL_FULL=1    — opt-in full im2col+GEMM for large float Conv2D (default 0 = partial).
#   NETKIT_LOOP_UNROLL=1    — EXPERIMENTAL: 4× unroll in reference kernels (default 0).
#                             Increases .text; verify flash headroom on MCU before use.
#
# Target architecture (empty = desktop CPU; sets ARM_MATH_* flags for CMSIS):
#   NETKIT_ARCH=CM4        — Cortex-M4 (ARM_MATH_CM4)
#   NETKIT_ARCH=CM7         — Cortex-M7 (ARM_MATH_CM7)
#   NETKIT_ARCH=M33        — Cortex-M33 (ARM_MATH_ARMV8MML)
#   NETKIT_ARCH=M55        — Cortex-M55 (ARM_MATH_M55)
#   NETKIT_ARCH=NEON       — Cortex-A with NEON (ARM_MATH_NEON)
#   See third_party/netkit_arch.mk for full list
#
# Common targets:
#   make              — cpu: netkit CLI + libnetkit.a (heap arena default)
#   make lib          — libnetkit.a for current NETKIT_TARGET
#   make build-all    — cpu: netkit + examples + C API tests; mcu/mpu: lib + examples
#   make test         — default regression: C++/C + fast Python (cpu only)
#   make test-full    — full regression incl. ONNX/backbone parity (manual / pre-release)
#   make test-cpp     — ./netkit test (cpu only)
#   make test-c       — ./tests/test_c_api (cpu only)
#   make embedded-smoke — lean MCU/MPU smoke binary
#   make test-embedded-smoke-matrix — MCU/MPU + NETKIT_ARCH + CMSIS profiles (host smoke)
#   make examples     — infer_cpp + infer_c
#   make export-mnist — regenerate MNIST model + cases (requires numpy)
#   make clean        — remove build products
#   make rebuild      — clean + make
#
# Examples:
#   make                                    # desktop (cpu, heap arena)
#   make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all
#   make NETKIT_TARGET=mcu lib              # lean runtime, global arena
#   make NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib

NETKIT_TARGET ?= cpu
NETKIT_GLOBAL_ARENA ?= 0
NETKIT_HEAP_ARENA ?= 0
NETKIT_WEIGHTS_IN_RAM ?= 0
ifeq ($(NETKIT_TARGET),cpu)
  NETKIT_CMSIS_DSP ?= 1
  NETKIT_CMSIS_NN ?= 0
else ifeq ($(NETKIT_TARGET),mcu)
  NETKIT_CMSIS_DSP ?= 1
  NETKIT_CMSIS_NN ?= 1
else ifeq ($(NETKIT_TARGET),mpu)
  NETKIT_CMSIS_DSP ?= 1
  NETKIT_CMSIS_NN ?= 0
else
  NETKIT_CMSIS_DSP ?= 0
  NETKIT_CMSIS_NN ?= 0
endif
NETKIT_IM2COL_FULL ?= 0
NETKIT_LOOP_UNROLL ?= 0
NETKIT_ARCH ?=

include third_party/netkit_arch.mk

# Host embedded smoke (desktop): portable CMSIS-DSP without CMSIS-Core headers.
ifeq ($(NETKIT_HOST_SMOKE),1)
  NETKIT_ARCH_CFLAGS += -D__GNUC_PYTHON__
endif

CC ?= clang
CXX ?= clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

RUNTIME_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/reference_kernel.cpp src/kernel_workspace.cpp src/cmsis_buffer_size.cpp src/ops.cpp \
                    src/conv2d.cpp src/depthwise_conv2d.cpp src/conv2d_layout.cpp src/conv_dispatch.cpp src/conv1x1_kernel.cpp src/conv_depthwise_kernel.cpp \
                    src/conv_direct_kernel.cpp src/im2col_partial.cpp src/im2col_full.cpp \
                    src/convnextv2_block.cpp src/mobilenetv4_uib.cpp src/resnet_basic_block.cpp src/yolox_decoupled_head.cpp src/mlp.cpp src/quant_ops.cpp src/quant_softmax_s8.cpp src/cmsis_dsp_util.cpp src/quant_trace.cpp src/cmsis_nn_quant_backend.cpp src/cmsis_quant_plan.cpp src/cnn_quant.cpp src/cnn.cpp \
                    src/layer_ops/nk_op_conv2d.cpp src/layer_ops/nk_op_depthwise_conv2d.cpp \
                    src/layer_ops/nk_op_convnextv2_block.cpp src/layer_ops/nk_op_mobilenetv4_uib.cpp src/layer_ops/nk_op_yolox_decoupled_head.cpp src/layer_ops/nk_op_resnet_basic_block.cpp src/layer_ops/nk_op_layernorm2d.cpp \
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
ifneq ($(wildcard $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c),)
  CMSIS_SOFTMAX_CFLAGS = -std=c11 -O2 -I$(CMSIS_NN_DIR)/Include
  CMSIS_SOFTMAX_SOURCES = \
    $(CMSIS_NN_DIR)/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c
  CMSIS_SOFTMAX_OBJECTS = $(CMSIS_SOFTMAX_SOURCES:$(CMSIS_NN_DIR)/%.c=build/cmsis_softmax/%.o)
  TARGET_CPPFLAGS += -DNETKIT_USE_CMSIS_SOFTMAX_S8=1 -I$(CMSIS_NN_DIR)/Include
endif
ifeq ($(NETKIT_CMSIS_NN),1)
  ifeq ($(NETKIT_TARGET),mcu)
    ifeq ($(NETKIT_ARCH_IS_M_PROFILE),1)
      NETKIT_CMSIS_NN_EFFECTIVE := 1
    else
      $(warning NETKIT_CMSIS_NN=1 ignored — set NETKIT_ARCH=CM4|M33|... (Cortex-M); using reference kernels)
    endif
  else
    $(warning NETKIT_CMSIS_NN=1 ignored on NETKIT_TARGET=$(NETKIT_TARGET) — CMSIS-NN is MCU + Cortex-M only; using reference kernels)
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

CMSIS_DSP_OBJECTS =
ifeq ($(NETKIT_CMSIS_DSP),1)
  CMSIS_DSP_DIR ?= third_party/CMSIS-DSP
  ifeq ($(wildcard $(CMSIS_DSP_DIR)/Include/arm_math.h),)
    $(error NETKIT_CMSIS_DSP=1 requires CMSIS-DSP at $(CMSIS_DSP_DIR) — run ./tools/fetch_cmsis_dsp.sh)
  endif
  include third_party/cmsis_dsp.mk
  TARGET_CPPFLAGS += -DNETKIT_USE_CMSIS_DSP=1 -I$(CMSIS_DSP_DIR)/Include -I$(CMSIS_DSP_DIR)/PrivateInclude
  RUNTIME_SOURCES += src/cmsis_dsp_backend.cpp
endif

DESKTOP_SOURCES = src/nk_regression.cpp src/cli.cpp src/test.cpp

ifeq ($(NETKIT_TARGET),cpu)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_CPU=1
  CORE_SOURCES = $(RUNTIME_SOURCES) $(DESKTOP_SOURCES)
  BUILD_CLI = 1
  BUILD_C_TESTS = 1
  ifeq ($(NETKIT_GLOBAL_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_GLOBAL_ARENA=1
  endif
else ifeq ($(NETKIT_TARGET),mcu)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MCU=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_HEAP_ARENA=1
  endif
else ifeq ($(NETKIT_TARGET),mpu)
  TARGET_CPPFLAGS += -DNETKIT_TARGET_MPU=1
  CORE_SOURCES = $(RUNTIME_SOURCES)
  BUILD_CLI = 0
  BUILD_C_TESTS = 0
  ifeq ($(NETKIT_HEAP_ARENA),1)
    TARGET_CPPFLAGS += -DNETKIT_HEAP_ARENA=1
  endif
else
  $(error NETKIT_TARGET must be cpu, mcu, or mpu (got '$(NETKIT_TARGET)'))
endif

TARGET_CPPFLAGS += -DNETKIT_WEIGHTS_IN_RAM=$(NETKIT_WEIGHTS_IN_RAM)
TARGET_CPPFLAGS += -DNETKIT_IM2COL_FULL=$(NETKIT_IM2COL_FULL)
TARGET_CPPFLAGS += -DNETKIT_LOOP_UNROLL=$(NETKIT_LOOP_UNROLL)

CFLAGS += $(TARGET_CPPFLAGS)
CXXFLAGS += $(TARGET_CPPFLAGS)

# GitHub Actions: unoptimized debug CPU builds make full-backbone regression very slow.
ifeq ($(GITHUB_ACTIONS),true)
  ifeq ($(NETKIT_TARGET),cpu)
    CFLAGS += -O2
    CXXFLAGS += -O2
  endif
endif

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
TRIM_RUNTIME_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/reference_kernel.cpp src/kernel_workspace.cpp src/cmsis_buffer_size.cpp src/ops.cpp \
                       src/conv2d.cpp src/conv2d_layout.cpp src/conv_dispatch.cpp src/conv1x1_kernel.cpp src/conv_depthwise_kernel.cpp \
                       src/conv_direct_kernel.cpp src/im2col_partial.cpp src/im2col_full.cpp src/mlp.cpp src/cnn.cpp $(TRIM_LAYER_OP_SOURCES) src/ops_resolver.cpp \
                       src/nk_format.cpp src/nk_loader.cpp src/netkit_api.cpp
TRIM_CORE_OBJECTS = $(TRIM_RUNTIME_SOURCES:.cpp=.o)

# Rebuild objects when target/backends change — avoids mixing CPU and MCU .o files in libnetkit.a.
NETKIT_BUILD_STAMP = .netkit_build_stamp
NETKIT_BUILD_ID = target=$(NETKIT_TARGET),global_arena=$(NETKIT_GLOBAL_ARENA),heap_arena=$(NETKIT_HEAP_ARENA),weights_in_ram=$(NETKIT_WEIGHTS_IN_RAM),cmsis_nn=$(NETKIT_CMSIS_NN_EFFECTIVE),cmsis_dsp=$(NETKIT_CMSIS_DSP),im2col_full=$(NETKIT_IM2COL_FULL),loop_unroll=$(NETKIT_LOOP_UNROLL),arch=$(NETKIT_ARCH),host_smoke=$(NETKIT_HOST_SMOKE)

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

.PHONY: all lib clean rebuild test test-full test-cpp test-c test-python test-python-full run example-c example-cpp examples \
        export-mnist export-mnist-int8 export-mnist-cnn export-mnist-all export-op-matrix \
        export-nk build-all embed-tests cmsis-nn-init cmsis-dsp-init cmsis-init \
        cpu cpu-global mcu mcu-heap mpu mpu-heap embedded-smoke test-embedded-smoke \
        test-embedded-smoke-matrix trim-lib check-trim-lib

ifeq ($(BUILD_CLI),1)
all: netkit-config-sync $(TARGET)
build-all: netkit-config-sync all examples $(TEST_C)
else
all: netkit-config-sync $(LIB)
build-all: netkit-config-sync $(LIB) examples embedded-smoke
endif

.DEFAULT_GOAL := all

$(LIB): $(CORE_OBJECTS) $(CMSIS_NN_OBJECTS) $(CMSIS_DSP_OBJECTS) $(CMSIS_SOFTMAX_OBJECTS)
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
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB)
endif

$(EXAMPLE_C): netkit-config-sync $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB)

$(EXAMPLE_CPP): netkit-config-sync $(LIB) $(EXAMPLE_CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_CPP_OBJ) $(LIB)

ifeq ($(BUILD_C_TESTS),1)
$(TEST_C): netkit-config-sync $(LIB) $(TEST_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_C_OBJ) $(LIB)
endif

$(EMBEDDED_SMOKE): netkit-config-sync $(LIB) $(EMBEDDED_SMOKE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EMBEDDED_SMOKE_OBJ) $(LIB)

ifeq ($(BUILD_CLI),1)
$(NK_INFER): netkit-config-sync $(LIB) $(NK_INFER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(NK_INFER_OBJ) $(LIB)
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
	rm -rf build/cmsis_nn build/cmsis_dsp build/cmsis_softmax

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

mcu:
	$(MAKE) NETKIT_TARGET=mcu lib

mcu-heap:
	$(MAKE) NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib

mpu:
	$(MAKE) NETKIT_TARGET=mpu lib

mpu-heap:
	$(MAKE) NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib

cmsis-nn-init:
	./tools/fetch_cmsis_nn.sh

cmsis-dsp-init:
	./tools/fetch_cmsis_dsp.sh

cmsis-core-init:
	./tools/fetch_cmsis_core.sh

cmsis-init: cmsis-nn-init cmsis-dsp-init cmsis-core-init

export-mnist:
	PYTHONPATH=python python3 tools/export_mnist_mlp.py

export-mnist-int8:
	PYTHONPATH=python python3 tools/export_mnist_mlp_int8.py

export-mnist-cnn-int8:
	PYTHONPATH=python python3 tools/export_mnist_cnn_int8.py --from-nk models/mnist_cnn.nk --fast
	python3 benchmark/tflm/tools/export_int8_test_images.py

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
