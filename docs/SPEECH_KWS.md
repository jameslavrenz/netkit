# Speech Commands KWS fixture (`speech_kws.nk`)

MCU-realistic keyword-spotting regression model with **fixed coefficients** and **embedded TCAS cases** (no JSON sidecars).

## Role

| Tier | Models | Typical size |
|------|--------|--------------|
| Hand / toy | `test_mlp`, `mlp_hand`, `cnn_hand` | &lt;1 KB |
| **Speech KWS** | **`speech_kws.nk`** | **~8 KB** |
| Desktop / MPU | MNIST, Fashion-MNIST | 440 KB–930 KB |

`speech_kws.nk` sits between hand fixtures and MNIST: a small conv stack on a compact MFCC-like feature map, sized for Cortex-M firmware bring-up and AOT arena probing.

## Model

- **Input:** `16 × 10 × 1` float feature map (downsampled time × mel bins)
- **Topology:** conv 3×3 (4) → max-pool 2×2 → conv 3×3 (8) → max-pool 2×2 → flatten → dense **12** (keyword logits)
- **Weights:** hand-authored in `tools/write_speech_kws_model.py` (not trained on Google Speech Commands audio)
- **Output:** 12 logits — `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`

Preprocessing (FFT / MFCC / framing) is **out of scope** for netkit. Tests use synthetic feature grids that mimic band energy and temporal onset patterns; embed your own precomputed features in TCAS when integrating on device.

## Regenerate

```bash
make export-speech-kws
# or
python3 tools/write_speech_kws_model.py
python3 tools/export_onnx_test_models.py   # writes speech_kws.onnx for ONNX parity
```

## Tests

| Harness | Cases |
|---------|------:|
| `make test-cpp` / `make test-c` | 8 embedded TCAS cases |
| `make test-python` (ONNX parity) | 8 (vs `speech_kws.onnx`) |
| `make test-python` (AOT compile) | included in `test_aot_compile.py` |

See [TESTING.md](TESTING.md).
