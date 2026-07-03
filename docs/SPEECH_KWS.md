# Speech Commands KWS (`speech_kws.nk`)

MCU-realistic keyword-spotting regression model with **trained coefficients** and **embedded TCAS cases** (no JSON sidecars).

## Role

| Tier | Models | Typical size |
|------|--------|--------------|
| Hand / toy | `test_mlp`, `mlp_hand`, `cnn_hand` | &lt;1 KB |
| **Speech KWS** | **`speech_kws.nk`** | **~13 KB** |
| Desktop / MPU | MNIST, Fashion-MNIST | 440 KB–930 KB |

`speech_kws.nk` sits between hand fixtures and MNIST: a depthwise-separable conv stack on compact mel features, sized for Cortex-M firmware bring-up and AOT arena probing.

## Model

- **Input:** `16 × 10 × 1` float feature map (downsampled time × mel bins)
- **Topology:** conv 1→8 (3×3, pad 1) → depthwise 8 (3×3, pad 1) → pointwise 8→16 (1×1) → max-pool 2×2 → depthwise 16 → pointwise 16→24 → max-pool 2×2 → flatten → dense **12** (keyword logits)
- **Weights:** trained on Google Speech Commands v0.02 (12-class subset) via `tools/export_speech_kws.py`
- **Test accuracy:** ~73% on held-out clips (800 samples/class, seed 42, 40 epochs)
- **Output:** 12 logits — `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`

Feature extraction (STFT → mel → log scale) lives in `python/netkit/speech_features.py` and is cached under `data/speech_commands/kws_16x10.npz` on first export. On device, embed your own precomputed `16×10` features in TCAS cases or feed them from your firmware preprocessor.

## Regenerate

Requires the Speech Commands tarball (downloaded automatically on first run; ~2.3 GB extracted under `data/speech_commands/`).

```bash
make export-speech-kws
# or
python3 tools/export_speech_kws.py
python3 tools/export_onnx_test_models.py   # writes speech_kws.onnx for ONNX parity
```

## Tests

| Harness | Cases |
|---------|------:|
| `make test-cpp` / `make test-c` | 12 embedded TCAS cases (one per keyword class) |
| `make test-python` (ONNX parity) | 12 (vs `speech_kws.onnx`) |
| `make test-python` (AOT compile) | included in `test_aot_compile.py` |
| `make test-embedded-smoke-matrix` | load/run silence features (160 inputs → 12 logits) on 7 MCU/MPU profiles |

Embedded smoke uses the **silence** TCAS case: all-zero mel features and trained-model logits — a realistic MCU workload without pulling in MNIST-sized weights.

See [TESTING.md](TESTING.md#embedded-smoke-mcupu).
