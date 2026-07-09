# XNNPACK (optional) — Google NN operator library for cpu/mpu LayerFast kernels.
#
# Fetch + build once:
#   ./tools/fetch_xnnpack.sh
#   # or: make xnnpack-init
#
# Expects:
#   third_party/XNNPACK/include/xnnpack.h
#   third_party/XNNPACK/netkit_lib/libXNNPACK.a (+ pthreadpool/cpuinfo/fxdiv)

XNNPACK_DIR ?= third_party/XNNPACK
XNNPACK_LIB_DIR ?= $(XNNPACK_DIR)/netkit_lib
