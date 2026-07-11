#!/usr/bin/env python3
"""CPU host A/B suite: netkit vs TF Lite — INT8.

Models: MNIST MLP int8, MNIST CNN int8, MNIST DW-CNN int8,
ImageNet MobileNetV4-Conv-Small int8.

Same fairness policy as host_ab_suite_common (prebuild, discard 1st process,
order swaps, equalized MLP runs, LiteRT-matched -O3).

Sweeps XNNPACK ON/OFF (reference when XNNPACK is off).
NETKIT_IM2COL is fixed at 0 (direct).
Also reports MCU-style runtime flash/RAM (ELF TEXT/DATA minus fixture images
vs LiteRT CPU libs; models excluded) with TF÷netkit ratios.
MLP uses batched invoke windows (1000×10) to escape ~1 µs timer noise.

Results default: benchmark/host_ab_suite_results_int8.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_ab_suite_common import INT8, cli_entry, ensure_assets_int8

if __name__ == "__main__":
    cli_entry(
        INT8,
        ensure_assets_int8,
        __doc__ or "Host A/B suite (int8)",
    )
