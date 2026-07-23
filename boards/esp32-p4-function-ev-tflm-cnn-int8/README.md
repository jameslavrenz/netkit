# ESP32-P4-Function-EV — MNIST CNN int8 (TFLM)

Uses Espressif’s [`esp-tflite-micro`](https://github.com/espressif/esp-tflite-micro) (TFLM + **ESP-NN**).

| Item | Value |
|------|--------|
| Method | **10×10**, discard first invoke each run |
| Arena | 96 KiB tensor arena |
| Peer | [`../esp32-p4-function-ev-cnn-int8/`](../esp32-p4-function-ev-cnn-int8/README.md) |

```bash
make -C ../../benchmark/tflm export-cnn-int8   # once
cd boards/esp32-p4-function-ev-tflm-cnn-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
```
