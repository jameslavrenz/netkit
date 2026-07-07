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

.PHONY: all fetch-tflm export-assets install-benchmark build build-mlp build-cnn build-cnn-profile build-mlp-profile run run-mlp run-cnn run-cnn-profile run-mlp-profile clean

all: run

fetch-tflm:
	@./tools/fetch_tflm.sh

export-assets:
	@python3 tools/export_assets.py

export-cnn-int8:
	@python3 tools/export_assets.py --model cnn-int8

export-int8-test-images:
	@python3 tools/export_int8_test_images.py

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

build: build-mlp build-cnn

run-mlp: build-mlp
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_mlp_benchmark

run-cnn: build-cnn
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_cnn_benchmark

run-cnn-profile: build-cnn-profile
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_cnn_profile_benchmark

run-mlp-profile: build-mlp-profile
	@$(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) EXTERNAL_DIR=$(EXTERNAL_REL) run_mnist_mlp_profile_benchmark

run: run-mlp run-cnn

clean:
	@if [ -f "$(TFLM_MAKE)" ]; then $(GMAKE) -C $(TFLM_ROOT) -f $(TFLM_MAKE) clean; fi
	@rm -rf generated/mlp_saved_model generated/cnn_saved_model generated/saved_model
