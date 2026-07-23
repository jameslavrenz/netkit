# esp32-p4-function-ev-cnn

P4 float32 netkit peer (MNIST CNN). ESP-NN off (no float API). Uses **lowered AOT**.

**Known issue (later):** float interpreter embed (`--no-lower`) mispredicts on this MCU (~2/10).
See [KNOWN_ISSUES.md KI-001](../../docs/KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device) (P4 and S3).
