# ESP32-P4-Function-EV — MNIST DS-CNN float32 (netkit reference, lowered AOT).
# Lowered is required on-device: interpreter embed float path mispredicts on P4.
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../esp32-p4-function-ev)
GEN_DIR := $(BOARD_ROOT)/generated
include $(BOARD_COMMON)/common.mk

NK_MODEL := $(NETKIT_ROOT)/models/mnist_cnn_dw.nk
AOT_STEM := mnist_cnn_dw

.PHONY: all check-deps export-embed build flash monitor run clean fullclean

all: check-deps export-embed build

check-deps:
	$(XIAO_CHECK_PIO)
	@$(PYTHON) -c 'import numpy, onnx' 2>/dev/null || \
		(echo "Python with numpy+onnx required" >&2; exit 1)
	@test -f $(NK_MODEL) || (echo "Missing $(NK_MODEL)" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc || \
		(echo "Missing float test images" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NK_MODEL)
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python $(PYTHON) -m netkit aot $(NK_MODEL) \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		--omit-final-softmax
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
