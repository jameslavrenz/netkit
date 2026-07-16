# NUCLEO-F446RE — MNIST DS-CNN int8 benchmark firmware

Bare-metal firmware for the **STM32 NUCLEO-F446RE** (STM32F446RET6, Cortex-M4F, 512 KiB flash / 128 KiB SRAM).

Runs the **MNIST DS-CNN (depthwise-separable)** int8 peer of `boards/nucleo-f446re-cnn-int8` (10 test images, 10 runs).

**Default build = quant lowered** (static `CmsisQuantPlan` + depthwise). Use `NETKIT_EMBED=1` for interpreter embed (TFLM-fair). Toggle reference kernels with `NETKIT_REFERENCE_QUANT_LOOPS=1`.

## netkit build profile (default)

| Setting | Value |
|---------|--------|
| Model | `models/mnist_cnn_dw_int8.nk` |
| Images | `benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.*` |
| Target | `NETKIT_TARGET_MCU_ARM` / `CM4` |
| CMSIS | **CMSIS-NN** (default); `NETKIT_REFERENCE_QUANT_LOOPS=1` for QuantOps |
| Deployment | **Quant lowered** (default); `NETKIT_EMBED=1` → interpreter embed |
| Arena | Tiny bump (lowered); **96 KiB** embed (`NETKIT_ARENA_KB=96`) |
| Dtype | int8 end-to-end; Softmax omitted; argmax logits |

## Build / flash / capture

```bash
cd boards/nucleo-f446re-cnn-dw-int8
make                          # quant lowered + CMSIS-NN
make NETKIT_EMBED=1           # interpreter embed (TFLM-fair)
make deploy-lowered           # clean + lowered rebuild
make NETKIT_REFERENCE_QUANT_LOOPS=1 clean all   # reference kernels
./scripts/flash.sh
PORT=/dev/cu.usbmodem11203 CAPTURE_SEC=180 ./scripts/deploy.sh capture
```

ELF: `build/mnist_cnn_dw_int8_nucleo_f446re.elf`

```text
BENCHMARK_SUMMARY runtime=netkit model=cnn_dw_int8 backend=cmsis-nn-int8 mean_us=... runs=10
```

## Verified on-device results (NUCLEO-F446RE @ 180 MHz, interpreter embed)

Matched −O2/−flto toolchain; 10×10, discard first invoke. See [STATUS.md](../../docs/STATUS.md) / [`mcu_ab_logs`](../../benchmark/mcu_ab_logs/).

| Mode | netkit (`NETKIT_EMBED=1`) | TFLM | microTVM |
|------|--------------------------:|-----:|---------:|
| CMSIS-NN | **58.3 ms** | 61.4 ms | 86.4 ms |
| reference (`NETKIT_REFERENCE_QUANT_LOOPS=1`) | **140.3 ms** | 826.8 ms | 236.0 ms |

All **10/10**. Peers: [tflm-cnn-dw-int8](../nucleo-f446re-tflm-cnn-dw-int8/README.md), [tvm-cnn-dw-int8](../nucleo-f446re-tvm-cnn-dw-int8/README.md).
