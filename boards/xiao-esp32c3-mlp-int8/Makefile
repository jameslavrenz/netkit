# Seeed Studio XIAO ESP32C3 — MNIST MLP int8 (ESP-IDF via PlatformIO).
#
# Usage:
#   make -C ../.. esp-nn-init export-mnist-mlp-int8   # once
#   make                                              # generate AOT + build
#   make flash monitor                                # USB Serial/JTAG
#   PORT=/dev/cu.usbmodemXXXX make flash

NETKIT_ROOT := $(abspath ../..)
BOARD_ROOT := $(abspath .)
GEN_DIR := $(BOARD_ROOT)/generated
PIO ?= $(HOME)/.platformio/penv/bin/pio
export PATH := $(dir $(PIO)):$(PATH)

NETKIT_LOWERED ?= 1
PORT ?=
# Prefer a python with numpy+onnx (AOT deps). Override with PYTHON=...
PYTHON ?= $(shell for p in python3 /usr/bin/python3 /opt/homebrew/bin/python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import numpy, onnx' 2>/dev/null && { command -v $$p; break; }; \
	done)

.PHONY: all check-deps export-embed build flash monitor run clean size

all: check-deps export-embed build

check-deps:
	@command -v $(notdir $(PIO)) >/dev/null 2>&1 || command -v pio >/dev/null 2>&1 || \
		(echo "PlatformIO required (pio). Install: https://platformio.org/install/cli" >&2; exit 1)
	@$(PYTHON) -c 'import numpy' 2>/dev/null || \
		(echo "Python with numpy required for AOT (set PYTHON=...)" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/third_party/ESP-NN/include/esp_nn.h || \
		(echo "Run: make -C $(NETKIT_ROOT) esp-nn-init" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/models/mnist_mlp_int8.nk || \
		(echo "Missing models/mnist_mlp_int8.nk — run: make -C $(NETKIT_ROOT) export-mnist-mlp-int8" >&2; exit 1)
	@test -f $(NETKIT_ROOT)/benchmark/tflm/generated/mnist_mlp_int8_test_images.cc || \
		(echo "Missing int8 test images — run: make -C $(NETKIT_ROOT) export-mnist-mlp-int8" >&2; exit 1)

export-embed: $(GEN_DIR)/.embed_stamp

$(GEN_DIR)/.embed_stamp: $(NETKIT_ROOT)/models/mnist_mlp_int8.nk
	@mkdir -p $(GEN_DIR)
	cd $(NETKIT_ROOT) && PYTHONPATH=$(NETKIT_ROOT)/python $(PYTHON) -m netkit aot models/mnist_mlp_int8.nk \
		-o $(abspath $(GEN_DIR)) --language cpp --target mcu --arena-headroom 0 \
		$(if $(filter 1,$(NETKIT_LOWERED)),--strict-lower,--no-lower) --omit-final-softmax
	@touch $(GEN_DIR)/.embed_stamp

build: export-embed
	@if [ -n "$(PORT)" ]; then export PORT="$(PORT)"; fi; \
	$(PIO) run -d $(BOARD_ROOT)

flash: export-embed
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--upload-port $(PORT)"; fi; \
	$(PIO) run -d $(BOARD_ROOT) -t upload $$PORT_ARG

monitor:
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--port $(PORT)"; fi; \
	$(PIO) device monitor -d $(BOARD_ROOT) -b 115200 $$PORT_ARG

run: flash
	@$(MAKE) monitor

size: build
	$(PIO) run -d $(BOARD_ROOT) -t size

clean:
	rm -rf $(BOARD_ROOT)/.pio $(BOARD_ROOT)/build
	rm -f $(GEN_DIR)/.embed_stamp
