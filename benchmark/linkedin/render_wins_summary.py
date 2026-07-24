#!/usr/bin/env python3
"""Render a LinkedIn wins card for MCU + MPU + host/CPU peers.

Peers covered:
  MCU  — TFLM (all boards); microTVM (NUCLEO-F446RE)
  MPU  — TF Lite / LiteRT (Raspberry Pi Zero 2 W)
  CPU  — TF Lite + ONNX Runtime (Apple M4 host)

Usage:
  benchmark/tflite/.venv/bin/python benchmark/linkedin/render_wins_summary.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "netkit_linkedin_wins.png"

W, H = 1920, 1080
BG = (22, 24, 29)
PANEL = (32, 35, 42)
PANEL2 = (28, 31, 38)
TEAL = (42, 157, 143)
TEAL_SOFT = (36, 72, 68)
AMBER = (212, 168, 75)
AMBER_SOFT = (72, 58, 32)
BLUE = (110, 150, 200)
BLUE_SOFT = (40, 52, 72)
WHITE = (236, 238, 242)
MUTED = (156, 163, 175)
LINE = (55, 60, 70)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    r: int = 14,
) -> None:
    draw.rounded_rectangle(box, radius=r, fill=fill)


def badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    f: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int],
    text: tuple[int, int, int],
) -> int:
    """Draw a small ISA/role badge; return right edge x."""
    x, y = xy
    pad_x, pad_y = 10, 4
    bbox = draw.textbbox((0, 0), label, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box = (x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y)
    draw.rounded_rectangle(box, radius=6, fill=fill)
    draw.text((x + pad_x, y + pad_y - 1), label, font=f, fill=text)
    return box[2]


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    f_title = font(44, bold=True)
    f_sub = font(20)
    f_brand = font(34, bold=True)
    f_hero = font(96, bold=True)
    f_hero_cap = font(24, bold=True)
    f_body = font(20)
    f_small = font(17)
    f_tiny = font(14)
    f_card_h = font(17, bold=True)
    f_num = font(32, bold=True)
    f_row = font(18)
    f_badge = font(13, bold=True)
    f_dev = font(18, bold=True)

    # Header
    d.text((56, 28), "netkit peer A/B  —  MCU · MPU · CPU", font=f_title, fill=WHITE)
    d.text(
        (56, 82),
        "Peers: TFLM  ·  microTVM  ·  TF Lite  ·  ONNX Runtime     ·     Fair A/B · Jul 2026 · warm latency · order swaps",
        font=f_sub,
        fill=MUTED,
    )
    d.text((W - 56 - 130, 34), "netkit", font=f_brand, fill=TEAL)

    # Legend for ISA badges
    lx = 56
    d.text((lx, 112), "ISA key:", font=f_tiny, fill=MUTED)
    lx = badge(d, (lx + 70, 108), "RISC-V", f_badge, fill=TEAL_SOFT, text=TEAL) + 10
    lx = badge(d, (lx, 108), "Arm", f_badge, fill=BLUE_SOFT, text=BLUE) + 10
    badge(d, (lx, 108), "Xtensa", f_badge, fill=AMBER_SOFT, text=AMBER)

    # Hero
    draw_rounded(d, (56, 145, 620, 430), PANEL)
    d.text((84, 160), "BIGGEST MCU WIN  ·  vs TFLM", font=f_card_h, fill=TEAL)
    d.text((84, 195), "9.9×", font=f_hero, fill=TEAL)
    d.text((84, 305), "faster than TFLM", font=f_hero_cap, fill=WHITE)
    d.text((84, 345), "XIAO ESP32-S3  ·  int8 reference", font=f_body, fill=WHITE)
    badge(d, (84, 375), "Xtensa", f_badge, fill=AMBER_SOFT, text=AMBER)
    d.text((84, 400), "MNIST CNN   netkit 112 ms  vs  TFLM 1113 ms", font=f_small, fill=MUTED)

    # Callouts — clearer device + ISA
    draw_rounded(d, (644, 145, 1220, 275), PANEL)
    d.text((668, 158), "vs microTVM", font=f_card_h, fill=TEAL)
    d.text((668, 185), "STM32 NUCLEO-F446RE", font=f_dev, fill=WHITE)
    badge(d, (668, 215), "Arm Cortex-M4F", f_badge, fill=BLUE_SOFT, text=BLUE)
    d.text((860, 212), "Up to 1.5×   ·   DS-CNN 58 vs 86 ms", font=f_small, fill=MUTED)

    draw_rounded(d, (1244, 145, 1864, 275), PANEL)
    d.text((1268, 158), "vs ONNX Runtime (XNNPACK EP)", font=f_card_h, fill=TEAL)
    d.text((1268, 185), "Mac mini · Apple M4", font=f_dev, fill=WHITE)
    badge(d, (1268, 215), "Arm", f_badge, fill=BLUE_SOFT, text=BLUE)
    d.text((1340, 212), "Up to 4.9× float  ·  2.5× int8", font=f_small, fill=MUTED)

    draw_rounded(d, (644, 295, 1220, 430), PANEL)
    d.text((668, 308), "vs TF Lite  ·  MPU", font=f_card_h, fill=TEAL)
    d.text((668, 335), "Raspberry Pi Zero 2 W", font=f_dev, fill=WHITE)
    badge(d, (668, 365), "Arm Cortex-A53", f_badge, fill=BLUE_SOFT, text=BLUE)
    d.text((860, 362), "Int8 ref up to 2.7×  ·  XNN ≈ parity", font=f_small, fill=MUTED)

    draw_rounded(d, (1244, 295, 1864, 430), PANEL)
    d.text((1268, 308), "HOST BINARY  ·  vs LiteRT", font=f_card_h, fill=TEAL)
    d.text((1268, 335), "Apple M4 host runtime image", font=f_dev, fill=WHITE)
    badge(d, (1268, 365), "Arm", f_badge, fill=BLUE_SOFT, text=BLUE)
    d.text((1340, 362), "~9× smaller  ·  1.3 vs 12.4 MiB", font=f_small, fill=MUTED)

    # MCU strip — full names + ISA
    d.text((56, 455), "MCU INT8 REFERENCE  (TFLM ÷ netkit)  —  portable kernels, vendor NN off", font=f_card_h, fill=MUTED)
    cards = [
        ("XIAO ESP32-S3", "Xtensa", "amber", "9.9×", "112 vs 1113 ms"),
        ("STM32 NUCLEO-F446RE", "Arm Cortex-M4F", "arm", "7.7×", "336 vs 2594 ms"),
        ("ESP32-P4-EV", "RISC-V", "riscv", "6.3×", "77 vs 485 ms"),
        ("XIAO ESP32-C3", "RISC-V", "riscv", "5.3×", "227 vs 1206 ms"),
    ]
    x0, cw, gap = 56, 444, 14
    for i, (name, isa, kind, gain, detail) in enumerate(cards):
        x = x0 + i * (cw + gap)
        draw_rounded(d, (x, 485, x + cw, 640), PANEL2)
        d.text((x + 20, 500), name, font=f_dev, fill=WHITE)
        if kind == "riscv":
            badge(d, (x + 20, 530), isa, f_badge, fill=TEAL_SOFT, text=TEAL)
        elif kind == "arm":
            badge(d, (x + 20, 530), isa, f_badge, fill=BLUE_SOFT, text=BLUE)
        else:
            badge(d, (x + 20, 530), isa, f_badge, fill=AMBER_SOFT, text=AMBER)
        d.text((x + 20, 568), gain, font=f_num, fill=TEAL)
        d.text((x + 20, 610), detail, font=f_small, fill=MUTED)

    # Bottom
    draw_rounded(d, (56, 665, 640, 915), PANEL)
    d.text((80, 682), "FLOAT32 MCU  ·  vs TFLM", font=f_card_h, fill=TEAL)
    d.text((80, 720), "ESP32-S3 (Xtensa): DS-CNN 2.6×", font=f_row, fill=WHITE)
    d.text((80, 752), "S3 (Xtensa) & P4 (RISC-V): CNN 1.7×", font=f_row, fill=WHITE)
    d.text((80, 792), "Lowered AOT · ESP-NN has no float API", font=f_small, fill=MUTED)
    d.text((80, 825), "All 10/10 top-1", font=f_small, fill=MUTED)
    d.text((80, 860), "ImageNet skipped on Espressif MCU (flash)", font=f_small, fill=TEAL)

    draw_rounded(d, (664, 665, 1220, 915), PANEL)
    d.text((688, 682), "VENDOR NN = PARITY  ·  vs TFLM", font=f_card_h, fill=TEAL)
    d.text((688, 720), "CMSIS-NN / ESP-NN ≈ 1.00–1.05×", font=f_row, fill=WHITE)
    d.text((688, 758), "Arm M4F · Xtensa S3 · RISC-V C3/P4", font=f_small, fill=MUTED)
    d.text((688, 792), "Same accel stack → same latency class", font=f_small, fill=MUTED)
    d.text((688, 826), "Also beats microTVM on NUCLEO (Arm)", font=f_small, fill=MUTED)
    d.text((688, 860), "NUCLEO flash/RAM leaner vs TFLM", font=f_small, fill=MUTED)

    draw_rounded(d, (1244, 665, 1864, 915), PANEL)
    d.text((1268, 682), "CPU  ·  Apple M4  ·  XNNPACK ON", font=f_card_h, fill=TEAL)
    d.text((1268, 720), "≈ TF Lite  ·  beats ORT on all six", font=f_row, fill=WHITE)
    d.text((1268, 758), "ORT÷nk up to 4.9× (float) / 2.5× (int8)", font=f_small, fill=MUTED)
    d.text((1268, 792), "vs TF Lite reference: up to ~4× (int8)", font=f_small, fill=MUTED)
    d.text((1268, 826), "Pi Zero (Arm A53) XNN also ≈ TF Lite", font=f_small, fill=MUTED)
    d.text((1268, 860), "MNIST + MobileNetV4-Small ImageNet", font=f_small, fill=MUTED)

    d.line((56, 940, W - 56, 940), fill=LINE, width=1)
    d.text(
        (56, 958),
        "RISC-V MCUs in this suite: ESP32-C3 · ESP32-P4  (still mcu_esp + ESP-NN, not mcu_risc)   ·   Arm: NUCLEO-F446RE · Pi Zero 2 W · Apple M4   ·   Xtensa: ESP32-S3",
        font=f_tiny,
        fill=MUTED,
    )
    d.text(
        (56, 988),
        "Sources: docs/STATUS.md  ·  Methodology: discard 1st invoke/process · order swaps · 1 thread MPU/CPU · 10×10 MCU  ·  Host ORT OFF uses MLAS (not a slow-ref peer)",
        font=f_tiny,
        fill=MUTED,
    )

    img.save(OUT, format="PNG", optimize=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
