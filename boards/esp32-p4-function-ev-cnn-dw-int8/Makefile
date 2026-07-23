# Seeed ESP32-P4-Function-EV — MNIST DS-CNN int8 (netkit ESP-NN, interpreter embed).
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../esp32-p4-function-ev)
GEN_DIR := $(BOARD_ROOT)/generated
include $(BOARD_COMMON)/common.mk

# TFLM-fair peer: always interpreter embed (CmsisQuantPlan → ESP-NN or QuantOps).
NETKIT_LOWERED := 0
NK_MODEL := $(NETKIT_ROOT)/models/mnist_cnn_dw_int8.nk
AOT_STEM := mnist_cnn_dw_int8

.PHONY: all check-deps export-embed build flash monitor run clean fullclean

all: check-deps export-embed build

check-deps:
	$(XIAO_CHECK_PIO)
	@$(PYTHON) -c 'import numpy, onnx' 2>/dev/null || \
		(echo "Python with numpy+onnx required (set PYTHON=...)" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/third_party/ESP-NN/include/esp_nn.h || \
		(echo "Run: make -C $(NETKIT_ROOT) esp-nn-init" >&2; exit 1)
	@test -f $(NK_MODEL) || \
		(echo "Missing $(NK_MODEL) — run: make -C $(NETKIT_ROOT) export-mnist-cnn-dw-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc || \
		(echo "Missing test images — run: make -C $(NETKIT_ROOT)/benchmark/tflm export-cnn-dw-int8" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NK_MODEL)
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python $(PYTHON) -m netkit aot $(NK_MODEL) \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		--no-lower --omit-final-softmax
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

fullclean:
	$(XIAO_PIO_FULLCLEAN)
	rm -rf $(BOARD_ROOT)/.pio
