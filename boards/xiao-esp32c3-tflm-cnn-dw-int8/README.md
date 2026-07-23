# XIAO ESP32C3 — MNIST DS-CNN int8 (TFLM)

Uses Espressif’s [`esp-tflite-micro`](https://github.com/espressif/esp-tflite-micro) (TFLM + **ESP-NN**).

| Item | Value |
|------|--------|
| Method | **10×10**, discard first invoke each run |
| Arena | 96 KiB tensor arena |
| Peer | [`../xiao-esp32c3-cnn-dw-int8/`](../xiao-esp32c3-cnn-dw-int8/README.md) |

```bash
make -C ../../benchmark/tflm export-cnn-dw-int8   # once
cd boards/xiao-esp32c3-tflm-cnn-dw-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
```
