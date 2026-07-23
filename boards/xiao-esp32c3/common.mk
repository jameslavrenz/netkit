# Shared Make helpers for boards/xiao-esp32c3-*
NETKIT_ROOT ?= $(abspath ../..)
BOARD_COMMON ?= $(abspath ../xiao-esp32c3)
PIO ?= $(HOME)/.platformio/penv/bin/pio
export PATH := $(dir $(PIO)):$(PATH)

PYTHON ?= $(shell for p in python3 /usr/bin/python3 /opt/homebrew/bin/python3 /opt/anaconda3/bin/python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import numpy, onnx' 2>/dev/null && { command -v $$p; break; }; \
	done)

PORT ?=

define XIAO_CHECK_PIO
	@command -v $(notdir $(PIO)) >/dev/null 2>&1 || command -v pio >/dev/null 2>&1 || \
		(echo "PlatformIO required (pio). Install: https://platformio.org/install/cli" >&2; exit 1)
endef

define XIAO_PIO_RUN
	$(PIO) run -d $(BOARD_ROOT)
endef

define XIAO_PIO_UPLOAD
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--upload-port $(PORT)"; fi; \
	$(PIO) run -d $(BOARD_ROOT) -t upload $$PORT_ARG
endef

define XIAO_PIO_MONITOR
	@PORT_ARG=""; \
	if [ -n "$(PORT)" ]; then PORT_ARG="--port $(PORT)"; fi; \
	$(PIO) device monitor -d $(BOARD_ROOT) -b 115200 $$PORT_ARG
endef
