# Peer-suite infographics

PNG summaries of completed inference A/B across **MCU** (vs TFLM), **MPU** (Pi Zero 2 W vs TF Lite), and **CPU** (Apple M4 vs TF Lite / ORT).

| File | Contents |
|------|----------|
| `netkit_linkedin_wins.png` | **LinkedIn one-panel** — MCU · MPU · CPU wins vs TFLM / microTVM / TF Lite / ONNX Runtime. Shown on [repo README](../../README.md#peer-benchmarks-mcu--mpu--cpu) and [STATUS — Peer highlights](../../docs/STATUS.md#peer-highlights-wins) |
| `netkit_linkedin_int8_suite.png` | Int8 latency (+ MCU flash/RAM) — shown on [repo README](../../README.md#peer-benchmarks-mcu--mpu--cpu) |
| `netkit_linkedin_float32_suite.png` | Float32 latency — shown on [repo README](../../README.md#peer-benchmarks-mcu--mpu--cpu) |
| `host_ab_linkedin_summary.png` | Older host-oriented LinkedIn summary card (not on README) |

Numbers and methodology: [docs/STATUS.md](../../docs/STATUS.md).

### Regenerate wins card

```bash
benchmark/tflite/.venv/bin/python benchmark/linkedin/render_wins_summary.py
# or any Python 3 with Pillow
```

Update the hardcoded latencies in `render_wins_summary.py` when STATUS peer tables change.
