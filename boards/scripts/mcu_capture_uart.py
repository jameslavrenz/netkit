#!/usr/bin/env python3
"""Flash-adjacent UART capture until BENCHMARK_SUMMARY/DONE or timeout."""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/cu.usbmodem11203")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pip install pyserial", file=sys.stderr)
        return 1

    deadline = time.time() + args.timeout
    buf = bytearray()
    with serial.Serial(args.port, args.baud, timeout=0.25) as ser:
        while time.time() < deadline:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)
                text = buf.decode("utf-8", errors="replace")
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                if "BENCHMARK_SUMMARY" in text and "DONE" in text:
                    break
            else:
                time.sleep(0.05)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(buf.decode("utf-8", errors="replace"))

    text = buf.decode("utf-8", errors="replace")
    if "BENCHMARK_SUMMARY" not in text:
        print("\nERROR: no BENCHMARK_SUMMARY in capture", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
