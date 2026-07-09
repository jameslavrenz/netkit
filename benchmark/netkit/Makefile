# netkit MNIST invoke benchmarks (pairs with benchmark/tflm/)
#
# Builds reference and CMSIS-DSP variants with TFLM-matched host flags.

ROOT := $(abspath ../..)

.PHONY: all run run-mlp run-cnn run-mlp-cmsis run-cnn-cmsis run-cmsis run-cnn-profile run-mlp-profile \
  run-mobilenetv4 run-mobilenetv4-int8 run-mobilenetv4-cmsis \
  run-aot run-aot-mlp run-aot-cnn run-aot-cmsis run-aot-mlp-cmsis run-aot-cnn-cmsis run-all clean clean-all

all: run

run-all: run run-cmsis

run: run-mlp run-cnn

run-cmsis: run-mlp-cmsis run-cnn-cmsis

run-mlp:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_BENCH=mnist_mlp_bench \
	  CNN_BENCH=mnist_cnn_bench \
	  MLP_MAIN_OBJ=src/main.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main.o \
	  run-mlp

run-cnn:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_BENCH=mnist_mlp_bench \
	  CNN_BENCH=mnist_cnn_bench \
	  MLP_MAIN_OBJ=src/main.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main.o \
	  run-cnn

run-mlp-unroll:
	$(MAKE) -f bench.mk BACKEND=loop-unroll CMSIS_DSP=0 NETKIT_LOOP_UNROLL=1 \
	  BENCH_OBJDIR=bench_obj_unroll \
	  BENCH_LIB=libnetkit_bench_unroll.a \
	  MLP_BENCH=mnist_mlp_bench_unroll \
	  CNN_BENCH=mnist_cnn_bench_unroll \
	  MLP_MAIN_OBJ=src/main_unroll.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_unroll.o \
	  run-mlp

run-cnn-unroll:
	$(MAKE) -f bench.mk BACKEND=loop-unroll CMSIS_DSP=0 NETKIT_LOOP_UNROLL=1 \
	  BENCH_OBJDIR=bench_obj_unroll \
	  BENCH_LIB=libnetkit_bench_unroll.a \
	  MLP_BENCH=mnist_mlp_bench_unroll \
	  CNN_BENCH=mnist_cnn_bench_unroll \
	  MLP_MAIN_OBJ=src/main_unroll.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_unroll.o \
	  run-cnn

run-mlp-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MLP_BENCH=mnist_mlp_bench_cmsis \
	  CNN_BENCH=mnist_cnn_bench_cmsis \
	  MLP_MAIN_OBJ=src/main_cmsis.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_cmsis.o \
	  run-mlp

run-cnn-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MLP_BENCH=mnist_mlp_bench_cmsis \
	  CNN_BENCH=mnist_cnn_bench_cmsis \
	  MLP_MAIN_OBJ=src/main_cmsis.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_cmsis.o \
	  run-cnn

# MobileNetV4-small float32 (host/CPU) — depthwise-heavy invoke benchmark.
run-mobilenetv4:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MNV4_BENCH=mobilenetv4_bench \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main.o \
	  run-mobilenetv4

run-mobilenetv4-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MNV4_BENCH=mobilenetv4_bench_cmsis \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main_cmsis.o \
	  run-mobilenetv4

run-mobilenetv4-int8:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MNV4_BENCH=mobilenetv4_bench \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main.o \
	  run-mobilenetv4-int8

run-cnn-profile:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  CNN_PROFILE_BENCH=mnist_cnn_profile_bench \
	  CNN_PROFILE_MAIN_OBJ=src/mnist_cnn_profile_main.o \
	  run-cnn-profile

run-mlp-profile:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_PROFILE_BENCH=mnist_mlp_profile_bench \
	  MLP_PROFILE_MAIN_OBJ=src/mnist_mlp_profile_main.o \
	  run-mlp-profile

run-aot: run-aot-mlp run-aot-cnn

run-aot-mlp:
	$(MAKE) -f aot_bench.mk run-mlp

run-aot-cnn:
	$(MAKE) -f aot_bench.mk run-cnn

run-aot-cmsis: run-aot-mlp-cmsis run-aot-cnn-cmsis

run-aot-mlp-cmsis:
	$(MAKE) -f aot_bench.mk run-mlp-cmsis

run-aot-cnn-cmsis:
	$(MAKE) -f aot_bench.mk run-cnn-cmsis

clean:
	rm -f mnist_mlp_bench mnist_cnn_bench mnist_cnn_profile_bench mnist_mlp_profile_bench \
	  mnist_mlp_bench_cmsis mnist_cnn_bench_cmsis \
	  mobilenetv4_bench mobilenetv4_bench_cmsis \
	  mnist_mlp_bench_aot mnist_cnn_bench_aot \
	  mnist_mlp_bench_aot_cmsis mnist_cnn_bench_aot_cmsis \
	  libnetkit_bench.a libnetkit_bench_cmsis.a \
	  src/main.o src/mnist_cnn_main.o src/mnist_cnn_profile_main.o src/mnist_mlp_profile_main.o \
	  src/main_cmsis.o src/mnist_cnn_main_cmsis.o \
	  src/mobilenetv4_main.o src/mobilenetv4_main_cmsis.o \
	  src/mnist_mlp_aot_main.o src/mnist_cnn_aot_main.o \
	  src/mnist_mlp_aot_main_cmsis.o src/mnist_cnn_aot_main_cmsis.o \
	  generated/aot/mnist_mlp_aot.o generated/aot/mnist_cnn_aot.o \
	  ../tflm/generated/mnist_test_images.o ../tflm/generated/mnist_cnn_test_images.o

clean-all: clean
	rm -rf bench_obj bench_obj_cmsis generated/aot
