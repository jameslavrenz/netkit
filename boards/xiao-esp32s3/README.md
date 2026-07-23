# ESP32-S3 MCU peers (netkit ↔ TFLM)

Board trees under `boards/xiao-esp32s3-*` target ESP32-S3 @ 240 MHz with USB Serial/JTAG.

The connected unit used for bring-up reports **16 MB PSRAM** and **32 MB OPI flash** (Espressif USB-JTAG `0x303a:0x1001`). PlatformIO still uses the `seeed_xiao_esp32s3` board id for the IDF/PIO recipe; flash header may say 8 MB while the chip is larger — that is OK for these firmwares.

## Flash / reset (important)

`esptool` hard-reset over USB-Serial/JTAG often reboots into **download mode** (`boot:0x0`) even when BOOT is not pressed (host DTR/RTS). Prefer OpenOCD builtin JTAG:

```bash
OCD=~/.platformio/packages/tool-openocd-esp32/bin/openocd
SCRIPTS=~/.platformio/packages/tool-openocd-esp32/share/openocd/scripts
FW=boards/xiao-esp32s3-cnn-int8/.pio/build/xiao_esp32s3
$OCD -s "$SCRIPTS" -f board/esp32s3-builtin.cfg \
  -c "program_esp $FW/bootloader.bin 0x0 verify" \
  -c "program_esp $FW/partitions.bin 0x8000 verify" \
  -c "program_esp $FW/firmware.bin 0x10000 verify reset exit"
```

Capture UART with DTR/RTS held inactive (do not pulse RTS).

## Board trees

| Tree | Model | Runtime | Notes |
|------|-------|---------|-------|
| `xiao-esp32s3-cnn-int8` | CNN int8 | netkit | ESP-NN S3 asm default; `PIO_ENV=xiao_esp32s3_ref` for QuantOps |
| `xiao-esp32s3-tflm-cnn-int8` | CNN int8 | TFLM | esp-nn / ref via `xiao_esp32s3_ref` |
| `xiao-esp32s3-cnn-dw-int8` | DS-CNN int8 | netkit | same as above |
| `xiao-esp32s3-tflm-cnn-dw-int8` | DS-CNN int8 | TFLM | same as above |
| `xiao-esp32s3-cnn` | CNN float32 | netkit | lowered AOT; ESP-NN N/A |
| `xiao-esp32s3-tflm-cnn` | CNN float32 | TFLM | reference float |
| `xiao-esp32s3-cnn-dw` | DS-CNN float32 | netkit | lowered AOT |
| `xiao-esp32s3-tflm-cnn-dw` | DS-CNN float32 | TFLM | reference float |

Float peers use a **4 MiB** factory partition (`partitions.csv`) and **SPIRAM** for the 256 KiB arena. **Lowered** float AOT is required — interpreter embed (`--no-lower`) mispredicts on S3 **and** P4 ([KI-001](../../docs/KNOWN_ISSUES.md#ki-001--espressif-mcu-float32-interpreter-embed-mispredicts-on-device)); S3 log: [`cnn_f32_netkit_embed.txt`](../../benchmark/mcu_ab_logs/xiao_esp32s3/cnn_f32_netkit_embed.txt) (2/10, many zero logits). Canonical summary: [`esp32s3_all_ab_results.txt`](../../benchmark/mcu_ab_logs/xiao_esp32s3/esp32s3_all_ab_results.txt) · [STATUS](../../docs/STATUS.md#mcu-seeed-xiao-esp32s3).

## ESP-NN (int8)

Default is **ESP32-S3 asm** (`NETKIT_ESP_NN_USE_S3_ASM=1`). Plan workspace is sized from `esp_nn_get_*_scratch_size` in `EspNnQuant::Finalize*Plan`. Disable with `-DNETKIT_ESP_NN_USE_S3_ASM=0` for portable opt.

## Smoke results (2026-07-23, 240 MHz, 10/10)

Logs: `benchmark/mcu_ab_logs/xiao_esp32s3/`.

### CNN int8

| Runtime | Backend | mean_us | mean_ms |
|---------|---------|--------:|--------:|
| netkit | ESP-NN S3 asm | 34720 | 34.720 |
| TFLM | esp-nn | 34851 | 34.851 |
| netkit | QuantOps reference | 112069 | 112.069 |
| TFLM | reference | 1113036 | 1113.036 |

### DS-CNN int8

| Runtime | Backend | mean_us | mean_ms |
|---------|---------|--------:|--------:|
| netkit | ESP-NN S3 asm | 31428 | 31.428 |
| TFLM | esp-nn | 31693 | 31.693 |
| netkit | QuantOps reference | 64307 | 64.307 |
| TFLM | reference | 362838 | 362.838 |

### CNN float32

| Runtime | Backend | mean_us | mean_ms |
|---------|---------|--------:|--------:|
| netkit | reference-f32 (lowered) | 308166 | 308.166 |
| TFLM | reference-f32 | 525624 | 525.624 |

### DS-CNN float32

| Runtime | Backend | mean_us | mean_ms |
|---------|---------|--------:|--------:|
| netkit | reference-f32 (lowered) | 63403 | 63.403 |
| TFLM | reference-f32 | 166353 | 166.353 |

## Quick smoke

```bash
make -C boards/xiao-esp32s3-cnn-int8 all
make -C boards/xiao-esp32s3-cnn all          # float needs SPIRAM + 4MiB partition
# flash via OpenOCD as above
```
