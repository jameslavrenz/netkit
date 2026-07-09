# TensorFlow Lite (LiteRT) host benches — MPU-oriented interpreter path.
#
# Distinct from benchmark/tflm/ (TensorFlow Lite Micro for MCUs).
#
# Fair comparison with netkit ImageNet benches:
#   make -C benchmark/netkit run-mobilenetv4-imagenet-xnnpack
# uses BENCH_FLAG_PROFILE=tflite (-O3 -DNDEBUG, SIMD on) from
# benchmark/common/tflite_host_flags.mk — not the TFLM Micro -O2 profile.

BENCH_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(BENCH_DIR)/../..)
VENV := $(BENCH_DIR)/.venv
PYTHON := $(VENV)/bin/python

.PHONY: venv run-mobilenetv4-imagenet run-mobilenetv4-imagenet-ref \
  run-mobilenetv4-imagenet-int8 run-mobilenetv4-imagenet-int8-ref clean

venv:
	@test -x $(PYTHON) || python3 -m venv $(VENV)
	@$(PYTHON) -c "import ai_edge_litert" 2>/dev/null || \
	  $(VENV)/bin/pip install -U pip ai-edge-litert pillow

# Default: XNNPACK CPU delegate (typical TF Lite MPU/desktop path), 1 thread.
run-mobilenetv4-imagenet: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet
	@test -d $(ROOT)/benchmark/tflm/generated/imagenet_sample_cache || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-imagenet-mnv4-images
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1

# Builtin reference kernels only (no XNNPACK) — closer to TFLM-style portable float.
run-mobilenetv4-imagenet-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1 --no-xnnpack

run-mobilenetv4-imagenet-int8: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet-int8
	@test -d $(ROOT)/benchmark/tflm/generated/imagenet_sample_cache || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-imagenet-mnv4-images
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1

run-mobilenetv4-imagenet-int8-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet-int8
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1 --no-xnnpack

clean:
	rm -rf $(VENV)
