# ESP32-P4-Function-EV — MNIST DS-CNN int8 (netkit)

| Item | Value |
|------|--------|
| Target | `mcu_esp` + `ESP32P4` + ESP-NN |
| Model | MNIST DS-CNN int8, interpreter embed |
| Arena | 96 KiB (MCU default is 64 KiB; DS-CNN embed needs more) |
| Method | **10×10**, discard first invoke each run (same as NUCLEO MCU peers) |
| Published | ESP-NN **87.7 ms** vs TFLM **87.5 ms**; reference **85.8 ms** vs TFLM **392.3 ms** (10/10) — [STATUS](../../docs/STATUS.md#mcu-seeed-esp32-p4-function-ev) |

```bash
make -C ../.. esp-nn-init
make -C ../../benchmark/tflm export-cnn-dw-int8   # once
cd boards/esp32-p4-function-ev-cnn-dw-int8 && make
PORT=/dev/cu.usbmodem* make flash monitor
# ESP-NN off (QuantOps reference):
PIO_ENV=esp32_p4_ev_ref make flash monitor
```

Peer vs TFLM: [`../esp32-p4-function-ev-tflm-cnn-dw-int8/`](../esp32-p4-function-ev-tflm-cnn-dw-int8/README.md).  
A/B runners: [`run_esp_int8_ab.sh`](../esp32-p4-function-ev/scripts/run_esp_int8_ab.sh) / [`run_esp_int8_ref_ab.sh`](../esp32-p4-function-ev/scripts/run_esp_int8_ref_ab.sh).
