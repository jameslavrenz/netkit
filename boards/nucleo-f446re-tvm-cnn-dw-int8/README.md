# NUCLEO-F446RE — microTVM AOT DS-CNN int8

Peer of [`nucleo-f446re-tvm-cnn-int8`](../nucleo-f446re-tvm-cnn-int8/README.md) for `mnist_cnn_dw_int8.tflite`.

## Fair-compare toolchain

Uses `boards/nucleo-f446re/mcu_tflm_toolchain.mk`:

- board glue: `MCU_CORE_*` (`-O2`)
- AOT codegen / CRT: `MCU_KERNEL_*` (`-O2`)
- CMSIS-NN: `MCU_THIRD_PARTY_KERNEL_*` (`-O2`)
- link: `-flto` + shared `STM32F446RETx_FLASH.ld` (with `__exidx_*`)

## Build

```bash
export TVM_HOME=/path/to/apache-tvm-v0.14
make compile-model              # CMSIS-NN BYOC
make
./scripts/flash.sh

make TVM_USE_CMSISNN=0 compile-model   # pure C AOT
make TVM_USE_CMSISNN=0 clean all
```

## On-device result (NUCLEO-F446RE @ 180 MHz)

| Runtime | Backend | Mean invoke | Accuracy |
|---------|---------|------------:|----------|
| microTVM | AOT CMSIS-NN | **86.4 ms** | 10/10 |
| microTVM | AOT C / reference | **236.0 ms** | 10/10 |
| netkit embed | CMSIS-NN | 58.3 ms | 10/10 |
| netkit embed | reference | 140.3 ms | 10/10 |
| TFLM | CMSIS-NN | 61.4 ms | 10/10 |
| TFLM | reference | 826.8 ms | 10/10 |

Full tables: [STATUS.md](../../docs/STATUS.md), [`mcu_ab_logs`](../../benchmark/mcu_ab_logs/).
