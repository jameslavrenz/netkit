# Known issues

Open bugs, measurement anomalies, and deferred follow-ups. Canonical tracker —
companion to [STATUS.md](STATUS.md) (maturity + published numbers) and
[API_PARITY.md](API_PARITY.md) (C ↔ C++ coverage).

When fixing an issue, remove or rewrite its entry here and update STATUS /
board READMEs in the same change.

---

## Bugs / correctness

### KI-001 — Espressif MCU float32 interpreter embed mispredicts on-device

| | |
|--|--|
| **Status** | Open — investigate later |
| **Severity** | Correctness on MCU float embed path |
| **Boards** | **ESP32-P4-Function-EV** and **XIAO ESP32-S3** (both confirmed) |
| **Symptom** | MNIST CNN float `--no-lower` (embedded `.nk` + `LoadCNNFromBuffer`) → ~**2/10** top-1; many images print **exact-zero** logits (S3 also showed sparse non-zeros / wrong argmax) |
| **Works** | Same `.nk` / embed path **10/10 on host**; **lowered AOT** float **10/10 on both boards**; **int8 embed** on both boards OK |
| **Workaround** | Published float peers use **lowered AOT** (no `--no-lower`). Int8 peers stay on interpreter embed |
| **Ruled out** | Arena size (256/272 KiB+), main stack 64 KiB, `-O3` / TFLM-match flags, flash XIP (RAM copy of blob), PSRAM + `esp_cache_msync` (P4) |
| **Next leads** | Float payload bind / `RepackConv2dWeights` (OIHW→HWIO, embed-only) vs lowered static `alignas(16)` weights; on-device weight CRCs vs host; force DRAM copy of float weights and/or disable HWIO repack. Softmax-omit for float CNN is a separate feature gap (see KI-005) and does not explain zeros (`Softmax(0)`→~0.1) |

#### Affected boards (side-by-side)

| Board | Chip / clock | Float embed (`--no-lower`) | Float lowered AOT | Int8 embed |
|-------|--------------|----------------------------|-------------------|------------|
| [ESP32-P4-Function-EV](../boards/esp32-p4-function-ev/README.md) | ESP32-P4 @ 360 MHz (RISC-V FPU) | **Fail** ~2/10; many zero logits; path ~2× too fast vs correct | **10/10** | **10/10** |
| [XIAO ESP32-S3](../boards/xiao-esp32s3/README.md) | ESP32-S3 @ 240 MHz (Xtensa FPU) | **Fail** 2/10; many zero logits (reproduced 2026-07-23) | **10/10** | **10/10** |

Not yet tried on XIAO ESP32C3 (no FPU float peer A/B). Treat as an **Espressif `mcu_esp` float-embed** bug until proven otherwise.

| Artifact | Board |
|----------|-------|
| [`esp32_p4_ev_float32_ab_results.txt`](../benchmark/mcu_ab_logs/esp32_p4_ev/esp32_p4_ev_float32_ab_results.txt) | P4 float peer (lowered) |
| [`cnn_f32_netkit_embed.txt`](../benchmark/mcu_ab_logs/xiao_esp32s3/cnn_f32_netkit_embed.txt) | S3 float embed failure log |
| [`esp32s3_all_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32s3/esp32s3_all_ab_results.txt) | S3 published peer summary |
| [STATUS — P4](STATUS.md#mcu-espressif-esp32-p4-function-ev) · [STATUS — S3](STATUS.md#mcu-seeed-xiao-esp32s3) | Published numbers |

**Note:** Interpreter embed remains the flexible path (reload `.nk` without re-lowering). BSS `ParsedModel` scratch does **not** remove that flexibility — only the float Espressif peer workaround uses lowered AOT.

---

## Measurement / performance anomalies

### KI-002 — XIAO ESP32C3: quant lowered AOT slower than interpreter embed

| | |
|--|--|
| **Status** | Open — investigate before promoting lowered as peer default |
| **Board** | Seeed XIAO ESP32C3 |
| **Symptom** | Under ESP-NN, quant lowered was a hair **slower** than interpreter embed (CNN ~254.6 vs 252.0 ms; DS-CNN ~88.5 vs 87.7 ms) |
| **Expected** | Lowered should win (static plan / weights, no `.nk` loader) |
| **Workaround** | C3 int8 peer boards stay on **interpreter embed** |
| **Refs** | [STATUS — C3](STATUS.md#mcu-seeed-xiao-esp32c3) · [MNIST_CNN.md](MNIST_CNN.md) |

### KI-003 — ESP32-P4: ESP-NN PIE asm disabled under PlatformIO gas

| | |
|--|--|
| **Status** | Open — revisit when toolchain accepts P4 immediates |
| **Symptom** | PIO `riscv32-esp` gas rejects some P4 PIE immediates (`esp.vldbc.8.ip …,1`) |
| **Workaround** | Peers use ESP-NN **portable / generic_opt**; opt-in later with `NETKIT_ESP_NN_USE_P4_ASM=1` |
| **Refs** | [boards/esp32-p4-function-ev/README.md](../boards/esp32-p4-function-ev/README.md) |

---

## Feature gaps (documented asymmetry)

### KI-004 — YOLOX detection accuracy

| | |
|--|--|
| **Status** | Open — more training / calibration |
| **Notes** | Runtime and host latency path land; mAP still needs work |
| **Refs** | [STATUS.md](STATUS.md) · [YOLOX.md](YOLOX.md) |

### KI-005 — Float CNN has no `omit_final_softmax` (C and C++)

| | |
|--|--|
| **Status** | Documented limitation (not a C-only hole) |
| **Notes** | MLP float + all int8 paths honor omit. Float CNN always runs Dense Softmax; `nk_cnn_set_omit_final_softmax` / quant runtime flag are **no-ops on float CNN**. Closing this requires a C++ float CNN omit API first, then C mirror |
| **Refs** | [API_PARITY.md](API_PARITY.md) · [c-api.md](c-api.md) · [cpp-api.md](cpp-api.md) |

### KI-006 — Deeper float AOT specialization (optional)

| | |
|--|--|
| **Status** | Optional enhancement |
| **Notes** | Fused/specialized codegen beyond calling shared `Kernels` / composite `::forward`; not required for correctness |

---

## Deferred (retrain / flash / platform TBD)

| ID | Item | Notes |
|----|------|-------|
| KI-D01 | Pi ImageNet int8 reference top-1 gap | netkit 7/10 vs TF Lite 8/10 under reference — retrain / recalibrate only; XNNPACK path already matches |
| KI-D02 | Float32 MNIST CNN / DS-CNN on NUCLEO-F446RE | Models exceed 512 KiB flash; use int8 on-device peers |
| KI-D03 | ESP32-S3 on-device peer A/B | **Done** 2026-07-23 — int8 + float32 CNN/DS-CNN; float stays lowered per KI-001 |
| KI-D03b | ESP32-C6 on-device peer A/B | Board trees optional / not published |
| KI-D04 | RISC-V MCU (`mcu_risc`) on-device peers | Runtime + host smoke done; Nuclei / RV32 vs TFLM / NMSIS-NN TBD |
| KI-D05 | Broader int8 model coverage | Beyond MNIST + ImageNet MNv4 fixtures |
| KI-D06 | float16 / int16 / int4 | Phase 2 — [DATATYPES.md](DATATYPES.md) |
| KI-D07 | Voice modality fixtures | Roadmap |

---

## Notes (not bugs)

- **MCU `ParsedModel` BSS scratch** — ~27 KiB catalog lives in static storage (not stack, not arena) so embed load does not smash FreeRTOS stacks. Still clears/reloads per `.nk`; does not force lowered AOT.
- **ImageNet on Espressif MCU peers** — skipped (weights / flash / arena budget vs MNIST peers).
- **C ↔ C++ API** — core runtime symbols in `netkit.h` have C++ counterparts; intentional C++-only helpers listed in [API_PARITY.md](API_PARITY.md#intentional-c-only-symbols). Keep docs in sync when adding APIs. Float CNN omit is a shared gap ([KI-005](#ki-005--float-cnn-has-no-omit_final_softmax-c-and-c)), not a missing C mirror.
