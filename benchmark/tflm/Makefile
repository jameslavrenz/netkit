# netkit TFLM MNIST invoke benchmarks

BENCH_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TFLM_ROOT := $(BENCH_DIR)/third_party/tflite-micro
TFLM_MAKE := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/Makefile
EXTERNAL_REL := ../..
GMAKE ?= $(shell command -v gmake 2>/dev/null || command -v gmak 2>/dev/null || true)
ifeq ($(GMAKE),)
  GMAKE := $(shell test -x /opt/homebrew/bin/gmake && echo /opt/homebrew/bin/gmake)
endif
ifeq ($(GMAKE),)
  GMAKE := make
endif

.PHONY: all fetch-tflm export-assets export-mobilenetv4 export-mobilenetv4-int8 \
  export-imagenet-mnv4-images export-mobilenetv4-imagenet export-mobilenetv4-imagenet-int8 \
  install-benchmark build build-mlp build-cnn build-cnn-profile build-mlp-profile \
  build-mobilenetv4 build-mobilenetv4-int8 build-mobilenetv4-imagenet \
  build-mobilenetv4-imagenet-int8 \
  run run-mlp run-cnn run-cnn-profile run-mlp-profile run-mobilenetv4 run-mobilenetv4-int8 \
  run-mobilenetv4-imagenet run-mobilenetv4-imagenet-int8 clean

all: run

fetch-tflm:
	@./tools/fetch_tflm.sh

export-assets:
	@python3 tools/export_assets.py

export-cnn-int8:
	@python3 tools/export_assets.py --model cnn-int8

export-mlp-int8:
	@python3 tools/export_assets.py --model mlp-int8

export-int8-test-images:
	@python3 tools/export_int8_test_images.py

export-mobilenetv4:
	@python3 tools/export_mobilenetv4.py

export-mobilenetv4-int8:
	@./tools/run_export_mobilenetv4_int8.sh
	@python3 tools/export_mobilenetv4_int8_test_images.py

export-imagenet-mnv4-images:
	@python3 tools/export_imagenet_mnv4_test_images.py

export-mobilenetv4-imagenet:
	@python3 tools/export_mobilenetv4_imagenet.py
	@$(MAKE) export-imagenet-mnv4-images

export-mobilenetv4-imagenet-int8:
	@test -d generated/imagenet_sample_cache || $(MAKE) export-imagenet-mnv4-images
	@./tools/run_export_mobilenetv4_imagenet_int8.sh
	@NETKIT_TF_VENV=$${NETKIT_TF_VENV:-/tmp/netkit-tf-venv}; \
	  if [ -x "$$NETKIT_TF_VENV/bin/python" ]; then \
	    $$NETKIT_TF_VENV/bin/python tools/export_imagenet_mnv4_int8_test_images.py; \
	  else \
	    python3 tools/export_imagenet_mnv4_int8_test_images.py; \
	  fi

install-benchmark: fetch-tflm
	@cp Makefile.inc $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/additional_tests.inc

build-mlp: install-benchmark
	@test -f generated/mnist_mlp.tflite && test -f generated/mnist_test_images.cc || $(MAKE) export-assets
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mnist_mlp_benchmark

build-cnn: install-benchmark
	@test -f generated/mnist_cnn.tflite && test -f generated/mnist_cnn_test_images.cc || $(MAKE) export-assets
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mnist_cnn_benchmark

build-cnn-profile: install-benchmark
	@test -f generated/mnist_cnn.tflite && test -f generated/mnist_cnn_test_images.cc || $(MAKE) export-assets
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mnist_cnn_profile_benchmark

build-mlp-profile: install-benchmark
	@test -f generated/mnist_mlp.tflite && test -f generated/mnist_test_images.cc || $(MAKE) export-assets
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mnist_mlp_profile_benchmark

build-mobilenetv4: install-benchmark
	@test -f generated/mobilenetv4_small.tflite || $(MAKE) export-mobilenetv4
	@test -f generated/mnist_cnn_test_images.cc || $(MAKE) export-assets
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mobilenetv4_benchmark

build-mobilenetv4-int8: install-benchmark
	@test -f generated/mobilenetv4_small_int8.tflite || $(MAKE) export-mobilenetv4-int8
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mobilenetv4_int8_benchmark

build-mobilenetv4-imagenet: install-benchmark
	@test -f generated/mobilenetv4_imagenet_f32.tflite || $(MAKE) export-mobilenetv4-imagenet
	@test -f generated/imagenet_mnv4_test_images.cc || $(MAKE) export-imagenet-mnv4-images
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mobilenetv4_imagenet_benchmark

build-mobilenetv4-imagenet-int8: install-benchmark
	@test -f generated/mobilenetv4_imagenet_int8.tflite || $(MAKE) export-mobilenetv4-imagenet-int8
	@test -f generated/imagenet_mnv4_int8_test_images.cc || $(MAKE) export-mobilenetv4-imagenet-int8
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) mobilenetv4_imagenet_int8_benchmark

build: build-mlp build-cnn

run-mlp: build-mlp
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_mlp_benchmark

run-cnn: build-cnn
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_cnn_benchmark

run-cnn-profile: build-cnn-profile
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_cnn_profile_benchmark

run-mlp-profile: build-mlp-profile
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_mlp_profile_benchmark

run-mobilenetv4: build-mobilenetv4
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mobilenetv4_benchmark

run-mobilenetv4-int8: build-mobilenetv4-int8
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mobilenetv4_int8_benchmark

run-mobilenetv4-imagenet: build-mobilenetv4-imagenet
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mobilenetv4_imagenet_benchmark

run-mobilenetv4-imagenet-int8: build-mobilenetv4-imagenet-int8
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mobilenetv4_imagenet_int8_benchmark

run: run-mlp run-cnn

clean:
	@if [ -f "$(TFLM_MAKE)" ]; then $(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) clean; fi
	@rm -rf generated/mlp_saved_model generated/cnn_saved_model generated/saved_model \
	  generated/mobilenetv4_imagenet_saved_model
