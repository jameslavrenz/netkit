# XIAO ESP32C3 — MNIST DS-CNN int8 (netkit)

| Item | Value |
|------|--------|
| Target | `mcu_esp` + `ESP32C3` + ESP-NN |
| Model | MNIST DS-CNN int8, quant lowered AOT |
| Method | **10×10**, discard first invoke each run (same as NUCLEO MCU peers) |

```bash
make -C ../.. esp-nn-init
make -C ../../benchmark/tflm export-cnn-dw-int8   # once
cd boards/xiao-esp32c3-cnn-dw-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
```

Peer vs TFLM: [`../xiao-esp32c3-tflm-cnn-dw-int8/`](../xiao-esp32c3-tflm-cnn-dw-int8/README.md).  
A/B runner (order swaps): [`../xiao-esp32c3/scripts/run_esp_int8_ab.sh`](../xiao-esp32c3/scripts/run_esp_int8_ab.sh).
