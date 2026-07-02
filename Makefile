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
# Optional CMSIS-NN kernels (Apache-2.0, fetch with ./tools/fetch_cmsis_nn.sh):
#   NETKIT_CMSIS_NN=1      — Cortex-M MCU only (NETKIT_TARGET=mcu + NETKIT_ARCH=CM4|M33|...)
#
# Optional CMSIS-DSP kernels (Apache-2.0, fetch with ./tools/fetch_cmsis_dsp.sh):
#   NETKIT_CMSIS_DSP=1     — use ARM CMSIS-DSP float32 vector/matrix ops in Ops::
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
#   make test         — full regression (cpu only)
#   make test-cpp     — ./netkit test (cpu only)
#   make test-c       — ./tests/test_c_api (cpu only)
#   make embedded-smoke — lean MCU/MPU smoke binary
#   make test-embedded-smoke-matrix — MCU/MPU + NETKIT_ARCH + CMSIS profiles
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
NETKIT_CMSIS_NN ?= 0
NETKIT_CMSIS_DSP ?= 0
NETKIT_ARCH ?=

include third_party/netkit_arch.mk

# Host embedded smoke (CI / desktop): portable CMSIS-DSP without CMSIS-Core headers.
ifeq ($(NETKIT_HOST_SMOKE),1)
  NETKIT_ARCH_CFLAGS += -D__GNUC_PYTHON__
endif

CC = clang
CXX = clang++
CFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c23 -Wall -Wextra -Iinclude
CXXFLAGS = -fcolor-diagnostics -fansi-escape-codes -g -std=c++26 -Wall -Wextra -Iinclude
TARGET = netkit
LIB = libnetkit.a

RUNTIME_SOURCES = src/arena.cpp src/tensor_factory.cpp src/tensor_access.cpp src/reference_kernel.cpp src/ops.cpp \
                    src/conv2d.cpp src/mlp.cpp src/cnn.cpp src/nk_format.cpp src/nk_loader.cpp \
                    src/netkit_api.cpp

TARGET_CPPFLAGS = $(NETKIT_ARCH_CFLAGS)

CMSIS_NN_OBJECTS =
NETKIT_CMSIS_NN_EFFECTIVE := 0
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

CFLAGS += $(TARGET_CPPFLAGS)
CXXFLAGS += $(TARGET_CPPFLAGS)

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

.PHONY: all lib clean rebuild test test-cpp test-c test-python run example-c example-cpp examples \
        export-mnist export-mnist-cnn export-mnist-all export-op-matrix \
        export-fashion-mnist export-fashion-mnist-cnn export-fashion-mnist-all \
        export-nk build-all embed-tests cmsis-nn-init cmsis-dsp-init cmsis-init \
        cpu cpu-global mcu mcu-heap mpu mpu-heap embedded-smoke test-embedded-smoke \
        test-embedded-smoke-matrix

ifeq ($(BUILD_CLI),1)
all: $(TARGET)
build-all: all examples $(TEST_C)
else
all: $(LIB)
build-all: $(LIB) examples embedded-smoke
endif

lib: $(LIB)

$(LIB): $(CORE_OBJECTS) $(CMSIS_NN_OBJECTS) $(CMSIS_DSP_OBJECTS)
	ar rcs $@ $^

ifeq ($(BUILD_CLI),1)
$(TARGET): $(LIB) $(CLI_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECTS) $(LIB)
endif

$(EXAMPLE_C): $(LIB) $(EXAMPLE_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_C_OBJ) $(LIB)

$(EXAMPLE_CPP): $(LIB) $(EXAMPLE_CPP_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EXAMPLE_CPP_OBJ) $(LIB)

ifeq ($(BUILD_C_TESTS),1)
$(TEST_C): $(LIB) $(TEST_C_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_C_OBJ) $(LIB)
endif

$(EMBEDDED_SMOKE): $(LIB) $(EMBEDDED_SMOKE_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(EMBEDDED_SMOKE_OBJ) $(LIB)

ifeq ($(BUILD_CLI),1)
$(NK_INFER): $(LIB) $(NK_INFER_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(NK_INFER_OBJ) $(LIB)
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXAMPLE_CPP_OBJ): $(EXAMPLE_CPP_SRC)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(NK_INFER_OBJ): $(NK_INFER_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

$(EXAMPLE_C_OBJ): $(EXAMPLE_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(BUILD_C_TESTS),1)
$(TEST_C_OBJ): $(TEST_C_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(EMBEDDED_SMOKE_OBJ): $(EMBEDDED_SMOKE_SRC) include/netkit.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJECTS) $(CLI_OBJECTS) $(EXAMPLE_C_OBJ) $(EXAMPLE_CPP_OBJ) $(TEST_C_OBJ) $(EMBEDDED_SMOKE_OBJ) $(NK_INFER_OBJ) \
	      $(TARGET) $(LIB) $(EXAMPLE_C) $(EXAMPLE_CPP) $(TEST_C) $(EMBEDDED_SMOKE) $(NK_INFER)
	rm -f src/*.o examples/*.o tests/*.o tools/*.o
	rm -rf build/cmsis_nn build/cmsis_dsp

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

test: test-cpp test-c test-python

test-python: $(TARGET) $(NK_INFER)
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

cmsis-init: cmsis-nn-init cmsis-dsp-init

export-mnist:
	PYTHONPATH=python python3 tools/export_mnist_mlp.py

export-mnist-cnn:
	PYTHONPATH=python python3 tools/export_mnist_cnn.py

export-mnist-all: export-mnist export-mnist-cnn
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-op-matrix:
	PYTHONPATH=python python3 tools/write_op_matrix_models.py
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-fashion-mnist:
	PYTHONPATH=python python3 tools/export_fashion_mnist_mlp.py
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-fashion-mnist-cnn:
	PYTHONPATH=python python3 tools/export_fashion_mnist_cnn.py
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-fashion-mnist-all: export-fashion-mnist export-fashion-mnist-cnn

export-onnx-test:
	PYTHONPATH=python python3 tools/export_onnx_test_models.py

export-nk:
	PYTHONPATH=python python3 -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk
	PYTHONPATH=python python3 -m netkit convert models/mlp_hand.onnx -o models/mlp_hand.nk
	PYTHONPATH=python python3 -m netkit convert models/test_cnn.onnx -o models/test_cnn.nk
	PYTHONPATH=python python3 -m netkit convert models/cnn_4x4_single.onnx -o models/cnn_4x4_single.nk
	PYTHONPATH=python python3 -m netkit convert models/cnn_hand.onnx -o models/cnn_hand.nk
	PYTHONPATH=python python3 -m netkit convert models/mnist_mlp.onnx -o models/mnist_mlp.nk
	PYTHONPATH=python python3 -m netkit convert models/mnist_cnn.onnx -o models/mnist_cnn.nk
	PYTHONPATH=python python3 tools/embed_nk_tests.py

embed-tests:
	PYTHONPATH=python python3 tools/embed_nk_tests.py
