"""Run bundled .nk models through the C++ runtime (tools/nk_infer)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def nk_infer_bin(root: Path | None = None) -> Path:
    base = root or ROOT
    env = os.environ.get("NK_INFER_BIN")
    if env:
        return Path(env)
    return base / "tools" / "nk_infer"


def run_nk_infer(nk_path: str | Path, flat_input: np.ndarray, *, root: Path | None = None) -> np.ndarray:
    """Forward one input through nk_infer; return flat float32 output."""
    bin_path = nk_infer_bin(root)
    if not bin_path.is_file():
        raise FileNotFoundError(f"nk_infer not found: {bin_path} (run make tools/nk_infer)")

    args = [str(bin_path), str(nk_path)] + [str(float(v)) for v in flat_input.reshape(-1)]
    proc = subprocess.run(
        args,
        cwd=root or ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nk_infer failed ({proc.returncode}):\n{proc.stderr}")
    body = proc.stdout.strip()
    if not body:
        return np.array([], dtype=np.float32)
    return np.array([float(v) for v in body.split(",")], dtype=np.float32)
