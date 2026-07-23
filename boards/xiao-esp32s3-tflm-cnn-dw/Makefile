# Seeed XIAO ESP32S3 — TFLM MNIST DS-CNN float32.
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../xiao-esp32s3)
GEN_DIR := $(BOARD_ROOT)/generated
include $(BOARD_COMMON)/common.mk

.PHONY: all check-deps export-model-data build flash monitor run clean fullclean

all: check-deps export-model-data build

check-deps:
	$(XIAO_CHECK_PIO)
	@test -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		$(BOARD_COMMON)/scripts/fetch_esp_tflite_micro.sh \
			$(NETKIT_ROOT)/boards/xiao-esp32c3-tflm-cnn-int8/components/esp-tflite-micro
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite || (echo "Missing tflite" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc || (echo "Missing float test images" >&2; exit 1)

export-model-data: $(GEN_DIR)/mnist_cnn_dw_model_data.cc

$(GEN_DIR)/mnist_cnn_dw_model_data.cc: $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite
	@mkdir -p $(GEN_DIR)
	$(PYTHON) $(BOARD_COMMON)/scripts/tflite_to_model_data.py $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite \
		--stem mnist_cnn_dw --array g_mnist_cnn_dw_model_data -o $(GEN_DIR)

build: check-deps export-model-data
	$(XIAO_PIO_RUN)

flash: export-model-data
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/managed_components $(BOARD_ROOT)/dependencies.lock $(GEN_DIR)

fullclean:
	$(XIAO_PIO_FULLCLEAN)
	rm -rf $(BOARD_ROOT)/.pio
