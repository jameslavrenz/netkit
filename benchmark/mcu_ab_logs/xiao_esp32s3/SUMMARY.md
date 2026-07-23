# XIAO ESP32-S3 MCU A/B smoke (2026-07-23)

Sysclk 240 MHz. Methodology: 10 runs × 10 images, discard first invoke each run. Published peer rows are 10/10 accuracy.

Canonical table: [`esp32s3_all_ab_results.txt`](esp32s3_all_ab_results.txt) · [STATUS](../../../docs/STATUS.md#mcu-seeed-xiao-esp32s3) · [KNOWN_ISSUES KI-001](../../../docs/KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device) (float embed fails on **S3 and P4**).

## CNN int8

| Runtime | Backend | mean_us |
|---------|---------|--------:|
| netkit | esp-nn-int8 (S3 asm) | 34720 |
| TFLM | esp-nn | 34851 |
| netkit | reference-int8 | 112069 |
| TFLM | reference | 1113036 |

## DS-CNN int8

| Runtime | Backend | mean_us |
|---------|---------|--------:|
| netkit | esp-nn-int8 (S3 asm) | 31428 |
| TFLM | esp-nn | 31693 |
| netkit | reference-int8 | 64307 |
| TFLM | reference | 362838 |

## CNN float32 (netkit lowered AOT)

| Runtime | Backend | mean_us |
|---------|---------|--------:|
| netkit | reference-f32 | 308166 |
| TFLM | reference-f32 | 525624 |

## DS-CNN float32 (netkit lowered AOT)

| Runtime | Backend | mean_us |
|---------|---------|--------:|
| netkit | reference-f32 | 63403 |
| TFLM | reference-f32 | 166353 |

## KI-001 probe (not a peer)

| Runtime | Path | Accuracy |
|---------|------|----------|
| netkit | float `--no-lower` embed | **2/10** (`cnn_f32_netkit_embed.txt`) |
