# XIAO ESP32S3 — MNIST CNN int8 (netkit)

| Item | Value |
|------|--------|
| Target | `mcu_esp` + `ESP32S3` + ESP-NN |
| Model | MNIST CNN int8, interpreter embed |
| Arena | 64 KiB (`NK_ARENA_DEFAULT_CAPACITY`) |
| Method | **10×10**, discard first invoke each run (same as NUCLEO MCU peers) |
| Published | ESP-NN **252.0 ms** vs TFLM **251.4 ms**; reference **226.8 ms** vs TFLM **1205.5 ms** (10/10) — [STATUS](../../docs/STATUS.md#mcu-seeed-xiao-esp32s3) |

```bash
make -C ../.. esp-nn-init
make -C ../../benchmark/tflm export-cnn-int8   # once
cd boards/xiao-esp32s3-cnn-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
# ESP-NN off (QuantOps reference):
PIO_ENV=xiao_esp32s3_ref make flash monitor
```

Peer vs TFLM: [`../xiao-esp32s3-tflm-cnn-int8/`](../xiao-esp32s3-tflm-cnn-int8/README.md).  
A/B runners: [`run_esp_int8_ab.sh`](../xiao-esp32s3/scripts/run_esp_int8_ab.sh) / [`run_esp_int8_ref_ab.sh`](../xiao-esp32s3/scripts/run_esp_int8_ref_ab.sh).
