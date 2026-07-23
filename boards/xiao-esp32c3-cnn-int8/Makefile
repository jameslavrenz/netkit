# Seeed XIAO ESP32C3 — MNIST CNN int8 (netkit ESP-NN, quant lowered AOT).
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../xiao-esp32c3)
GEN_DIR := $(BOARD_ROOT)/generated
include $(BOARD_COMMON)/common.mk

NETKIT_LOWERED ?= 1
NK_MODEL := $(NETKIT_ROOT)/models/mnist_cnn_int8.nk
AOT_STEM := mnist_cnn_int8

.PHONY: all check-deps export-embed build flash monitor run clean

all: check-deps export-embed build

check-deps:
	$(XIAO_CHECK_PIO)
	@$(PYTHON) -c 'import numpy, onnx' 2>/dev/null || \
		(echo "Python with numpy+onnx required (set PYTHON=...)" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/third_party/ESP-NN/include/esp_nn.h || \
		(echo "Run: make -C $(NETKIT_ROOT) esp-nn-init" >&2; exit 1)
	@test -f $(NK_MODEL) || \
		(echo "Missing $(NK_MODEL) — run: make -C $(NETKIT_ROOT) export-mnist-cnn-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc || \
		(echo "Missing test images — run: make -C $(NETKIT_ROOT)/benchmark/tflm export-cnn-int8" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NK_MODEL)
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python $(PYTHON) -m netkit aot $(NK_MODEL) \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		$(if $(filter 1,$(NETKIT_LOWERED)),--strict-lower,--no-lower) --omit-final-softmax
	@touch $(GEN_DIR)/.embed_stamp

build: export-embed
	$(XIAO_PIO_RUN)

flash: export-embed
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio
	rm -f $(GEN_DIR)/.embed_stamp
