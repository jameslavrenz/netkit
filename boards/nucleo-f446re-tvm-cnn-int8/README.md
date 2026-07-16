# NUCLEO-F446RE — microTVM AOT MNIST CNN int8

Bare-metal microTVM ahead-of-time (C runtime) peer for the netkit / TFLM MNIST CNN int8 boards.

## Prerequisites

1. Relay-era TVM **v0.14** built with `USE_MICRO=ON` and (for CMSIS path) `USE_CMSISNN=ON`:

   ```bash
   export TVM_HOME=$HOME/workspace/apache-tvm-v0.14
   ```

2. Conda env `tvm014` (Python 3.10 + NumPy 1.x) — see board `Makefile`.

3. Shared TFLite model + test images:

   ```bash
   make -C ../../benchmark/tflm export-cnn-int8
   ```

## Fair-compare toolchain

Uses `boards/nucleo-f446re/mcu_tflm_toolchain.mk` (same as TFLM/netkit peers):

- board glue: `MCU_CORE_*` (`-O2`)
- AOT codegen / CRT: `MCU_KERNEL_*` (`-O2`)
- CMSIS-NN: `MCU_THIRD_PARTY_KERNEL_*` (`-O2`)
- link: `-flto` + shared `STM32F446RETx_FLASH.ld` (with `__exidx_*`)

## Build / flash

```bash
export TVM_HOME=/path/to/apache-tvm-v0.14
make compile-model              # CMSIS-NN BYOC (default)
make
./scripts/flash.sh

# Pure C AOT (no CMSIS-NN):
make TVM_USE_CMSISNN=0 compile-model
make TVM_USE_CMSISNN=0 clean all
```

## Notes

- Same 10 MNIST images and timing methodology as `nucleo-f446re-cnn-int8` / `nucleo-f446re-tflm-cnn-int8`.
- PyPI `apache-tvm` 0.25+ dropped Relay/microTVM; this board expects a local v0.14 build.
- Apply `patches/cmsisnn_skip_per_channel_fc.patch` to TVM so per-channel `qnn.dense` stays on the C path (TVM 0.14 CMSIS-NN FC expects scalar scales). Conv / max-pool / softmax still go to CMSIS-NN when `TVM_USE_CMSISNN=1`.
- DS-CNN peer: [`nucleo-f446re-tvm-cnn-dw-int8`](../nucleo-f446re-tvm-cnn-dw-int8/README.md).

## On-device result (NUCLEO-F446RE @ 180 MHz, matched flags)

| Runtime | Backend | Mean invoke | Accuracy |
|---------|---------|------------:|----------|
| microTVM (this board) | AOT CMSIS-NN (conv/pool/softmax; FC in C) | **112.3 ms** | 10/10 |
| microTVM (this board) | AOT C / reference (`TVM_USE_CMSISNN=0`) | **343.0 ms** | 10/10 |
| netkit | CMSIS-NN interpreter embed (`NETKIT_EMBED=1`) | 95.3 ms | 10/10 |
| netkit | reference embed | 336.2 ms | 10/10 |
| TFLM | CMSIS-NN | 95.5 ms | 10/10 |
| TFLM | reference | 2593.5 ms | 10/10 |

UART logs: [`benchmark/mcu_ab_logs/tvm_cnn_int8_*.log`](../../benchmark/mcu_ab_logs/).
