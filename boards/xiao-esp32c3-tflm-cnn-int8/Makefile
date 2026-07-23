# Seeed XIAO ESP32C3 — TFLM MNIST CNN int8 (esp-tflite-micro + ESP-NN).
NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
BOARD_COMMON := $(abspath ../xiao-esp32c3)
include $(BOARD_COMMON)/common.mk

.PHONY: all check-deps build flash monitor run clean

all: check-deps build

check-deps:
	$(XIAO_CHECK_PIO)
	@test -f $(BOARD_ROOT)/components/esp-tflite-micro/CMakeLists.txt || \
		$(BOARD_COMMON)/scripts/fetch_esp_tflite_micro.sh $(BOARD_ROOT)/components/esp-tflite-micro
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_model_data.cc || \
		(echo "Missing model data — run: make -C $(NETKIT_ROOT)/benchmark/tflm export-cnn-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_cnn_int8_test_images.cc || \
		(echo "Missing test images" >&2; exit 1)

build:
	$(XIAO_PIO_RUN)

flash:
	$(XIAO_PIO_UPLOAD)

monitor:
	$(XIAO_PIO_MONITOR)

run: flash
	@$(MAKE) monitor

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/managed_components $(BOARD_ROOT)/dependencies.lock
