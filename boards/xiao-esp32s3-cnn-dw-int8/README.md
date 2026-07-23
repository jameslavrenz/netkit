# XIAO ESP32S3 — MNIST DS-CNN int8 (netkit)

| Item | Value |
|------|--------|
| Target | `mcu_esp` + `ESP32S3` + ESP-NN |
| Model | MNIST DS-CNN int8, interpreter embed |
| Arena | 96 KiB (MCU default is 64 KiB; DS-CNN embed needs more) |
| Method | **10×10**, discard first invoke each run (same as NUCLEO MCU peers) |
| Published | ESP-NN **87.7 ms** vs TFLM **87.5 ms**; reference **85.8 ms** vs TFLM **392.3 ms** (10/10) — [STATUS](../../docs/STATUS.md#mcu-seeed-xiao-esp32s3) |

```bash
make -C ../.. esp-nn-init
make -C ../../benchmark/tflm export-cnn-dw-int8   # once
cd boards/xiao-esp32s3-cnn-dw-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
# ESP-NN off (QuantOps reference):
PIO_ENV=xiao_esp32s3_ref make flash monitor
```

Peer vs TFLM: [`../xiao-esp32s3-tflm-cnn-dw-int8/`](../xiao-esp32s3-tflm-cnn-dw-int8/README.md).  
A/B runners: [`run_esp_int8_ab.sh`](../xiao-esp32s3/scripts/run_esp_int8_ab.sh) / [`run_esp_int8_ref_ab.sh`](../xiao-esp32s3/scripts/run_esp_int8_ref_ab.sh).
