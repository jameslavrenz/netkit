# esp32-p4-function-ev-cnn-dw

P4 float32 netkit peer (MNIST DS-CNN). ESP-NN off (no float API). Uses **lowered AOT**.

**Known issue (later):** float interpreter embed (`--no-lower`) mispredicts on this MCU (~2/10).
See [KNOWN_ISSUES.md KI-001](../../docs/KNOWN_ISSUES.md#ki-001--esp32-p4-float32-interpreter-embed-mispredicts-on-device).
