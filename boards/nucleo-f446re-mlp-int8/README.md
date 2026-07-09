# NUCLEO-F446RE — MNIST MLP int8 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **same MNIST MLP benchmark** as `benchmark/netkit/` and `benchmark/tflm/` (10 test images, 10 runs), using **int8** weights, activations, and prequantized inputs end-to-end.

**Default build = interpreter embed** (embedded `.nk` + runtime loader) for a fair comparison with TFLM `MicroInterpreter`.

## netkit build profile (default)

| Setting | Value |
|---------|--------|
| Target | `NETKIT_TARGET_MCU` |
| Arch | `NETKIT_ARCH=CM4` (Cortex-M4F) |
| CMSIS | **CMSIS-NN** + **CMSIS-DSP** (int8 FC; Softmax omitted — classify via argmax on logits) |
| Weights | **Flash** — embedded `.nk` blob in `.rodata` (`NETKIT_WEIGHTS_IN_RAM=0`) |
| Deployment | **Interpreter embed** — `NkLoader` + quantized MLP forward |
| Dtype | int8 weights / activations; prequantized int8 test inputs; output = logits (Softmax omitted) |

Set `NETKIT_REFERENCE_QUANT_LOOPS=1` to benchmark reference quant loops instead of CMSIS-NN kernels.

## Verified on-device results (NUCLEO-F446RE @ 180 MHz)

| Backend | Mean invoke | Accuracy |
|---------|------------:|----------|
| CMSIS-NN (`NETKIT_REFERENCE_QUANT_LOOPS=0`) | **~3.36 ms** | **10/10** |
| Reference (`NETKIT_REFERENCE_QUANT_LOOPS=1`) | **~15.0 ms** | **10/10** |

Compare with TFLM int8 on the same board: [nucleo-f446re-tflm-mlp-int8](../nucleo-f446re-tflm-mlp-int8/README.md).

## Build

```bash
# Prerequisites (once, from repo root)
make export-mnist-mlp-int8
../nucleo-f446re/scripts/setup-toolchain.sh

cd boards/nucleo-f446re-mlp-int8
make                                    # CMSIS-NN (default)
make NETKIT_REFERENCE_QUANT_LOOPS=1     # reference quant loops
./scripts/flash.sh
./scripts/monitor.sh   # press RESET on board
```

**UART capture tip:** open the ST-Link VCP **before** flash/reset, or use `./scripts/deploy.sh capture` after a manual reset — otherwise boot output can be missed.

## `BENCHMARK_SUMMARY` line

```text
BENCHMARK_SUMMARY runtime=netkit model=mlp_int8 backend=cmsis-nn-int8 mean_us=... runs=10
BENCHMARK_SUMMARY runtime=netkit model=mlp_int8 backend=reference-int8 mean_us=... runs=10
```
