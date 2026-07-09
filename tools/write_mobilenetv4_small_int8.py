#!/usr/bin/env python3
"""Generate models/mobilenetv4_small_int8.nk from the float32 MobileNetV4-small fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantize_float_input, quantized_cnn_to_spec
from netkit.reader import read_nk, read_test_suite
from netkit.writer import RegressionCase, RegressionSuite, write_nk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "mobilenetv4_small.nk"
OUT = ROOT / "models" / "mobilenetv4_small_int8.nk"


def main() -> None:
    arch, weights = read_nk(SRC)
    suite = read_test_suite(SRC)

    cal = []
    for case in suite.cases:
        cal.append(np.asarray(case.input, dtype=np.float32))
    cal_arr = np.stack(cal, axis=0)

    pack = quantize_cnn(arch, weights, cal_arr, num_calibration=len(cal))
    spec = quantized_cnn_to_spec(arch, pack)

    input_scale = pack.quant_layers[0].input_scale
    input_zp = pack.quant_layers[0].input_zero_point
    int8_cases = []
    for case in suite.cases:
        inp = np.asarray(case.input, dtype=np.float32)
        out_f = forward_quantized_cnn(inp, arch, pack, output_float=True)
        # TCAS stores floats; embed prequantized int8 as float values in [-128, 127].
        input_i8 = quantize_float_input(inp.reshape(-1), input_scale, input_zp)
        int8_cases.append(
            RegressionCase(
                name=case.name + " (int8)",
                input=input_i8.astype(np.float32).tolist(),
                expected=out_f.astype(np.float32).tolist(),
            )
        )

    spec.tests = RegressionSuite(tolerance=2.0, cases=int8_cases)
    write_nk(OUT, spec)
    print(f"Wrote {OUT} ({len(pack.weight_tensors)} weight tensors, {len(int8_cases)} TCAS cases)")


if __name__ == "__main__":
    main()
