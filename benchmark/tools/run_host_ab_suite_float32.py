#!/usr/bin/env python3
"""CPU host A/B suite: netkit vs TF Lite — FLOAT32.

Models: MNIST MLP, MNIST CNN, MNIST DW-CNN, ImageNet MobileNetV4-Conv-Small.

Same fairness policy as host_ab_suite_common (prebuild, discard 1st process,
order swaps, equalized MLP runs, LiteRT-matched -O3).

Sweeps XNNPACK ON/OFF (reference when XNNPACK is off).
NETKIT_IM2COL is fixed at 0 (direct).
Also reports MCU-style runtime flash/RAM (ELF TEXT/DATA minus fixture images
vs LiteRT CPU libs; models excluded) with TF÷netkit ratios.
MLP uses batched invoke windows (1000×10) to escape ~1 µs timer noise.

Results default: benchmark/host_ab_suite_results_float32.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_ab_suite_common import FLOAT32, cli_entry, ensure_assets_float32

if __name__ == "__main__":
    cli_entry(
        FLOAT32,
        ensure_assets_float32,
        __doc__ or "Host A/B suite (float32)",
    )
