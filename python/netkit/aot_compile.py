"""Ahead-of-time compiler: embed a .nk model as C or C++ source for the netkit runtime."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np

from .arch_writer import arch_to_nk_bytes
from .format import FLAG_HAS_QUANT, HEADER_BYTES, unpack_header, weight_payload_bytes
from .nk_optimize import OptimizeOptions, optimize_nk
from .aot_lower import (
    can_lower_arch,
    plan_lowered,
    render_lowered_cpp_header,
    render_lowered_cpp_source,
)
from .aot_lower_quant import (
    can_lower_quantized_arch,
    plan_lowered_quant,
    render_lowered_quant_cpp_header,
    render_lowered_quant_cpp_source,
)
from .reader import read_nk, read_test_suite
from .reference_forward import forward_cnn, forward_mlp


class AotLanguage(str, Enum):
    """Output language for generated sources."""

    CPP = "cpp"
    C = "c"


@dataclass(frozen=True)
class AotCompileResult:
    """Paths and metadata for a completed AOT compile."""

    model_name: str
    language: AotLanguage
    header_path: Path
    source_path: Path
    input_elements: int
    output_elements: int
    network: str
    nk_bytes: int
    arena_bytes_after_load: int
    arena_bytes_after_forward: int
    arena_bytes_recommended: int
    optimized: bool = False
    optimizations_applied: tuple[str, ...] = ()
    lowered: bool = False
    quant_fast: bool = False


def _sanitize_symbol(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned.lower()


def _compute_io(arch: dict, weights: np.ndarray) -> tuple[int, int]:
    network = arch["network"]
    if network == "mlp":
        batch, features = arch["input"]
        flat_input = np.zeros(batch * features, dtype=np.float32)
        output = forward_mlp(flat_input, arch, weights)
    elif network == "cnn":
        height, width, channels = arch["input"]
        flat_input = np.zeros(height * width * channels, dtype=np.float32)
        output = forward_cnn(flat_input, arch, weights)
    else:
        raise ValueError(f"unsupported network kind: {network}")
    output_arr = np.asarray(output, dtype=np.float32)
    return int(flat_input.size), int(output_arr.size)




def _io_elements_from_arch(arch: dict) -> tuple[int, int]:
    network = arch["network"]
    if network == "mlp":
        batch, features = arch["input"]
        input_elements = int(batch * features)
    elif network == "cnn":
        height, width, channels = arch["input"]
        input_elements = int(height * width * channels)
    else:
        raise ValueError(f"unsupported network kind: {network}")

    last = arch["layers"][-1]
    if last["type"] != "dense":
        raise ValueError(f"unsupported output layer for IO sizing: {last['type']}")
    return input_elements, int(last["units"])

def _format_byte_array(data: bytes, indent: str = "    ", width: int = 12) -> str:
    lines: list[str] = []
    row: list[str] = []
    for index, byte in enumerate(data):
        row.append(f"0x{byte:02x}")
        if len(row) == width:
            lines.append(f"{indent}{', '.join(row)},")
            row = []
    if row:
        lines.append(f"{indent}{', '.join(row)},")
    return "\n".join(lines)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _recommend_arena_bytes(after_forward: int, *, headroom_percent: int) -> int:
    headroom = max(0, headroom_percent)
    bumped = after_forward + (after_forward * headroom + 99) // 100
    return _round_up(bumped, 64)


def _repo_root_for_nk(nk_path: Path) -> Path | None:
    for parent in nk_path.resolve().parents:
        if (parent / "Makefile").is_file():
            return parent
    return None


def _ensure_netkit_binary(nk_path: Path) -> Path | None:
    """Locate ./netkit for inspect --full probing; build it in-repo when missing."""
    found = shutil.which("netkit")
    if found:
        return Path(found)

    root = _repo_root_for_nk(nk_path)
    if root is None:
        return None

    candidate = root / "netkit"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate

    subprocess.run(
        ["make", "netkit"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _find_netkit_binary(nk_path: Path) -> Path | None:
    return _ensure_netkit_binary(nk_path)


def _estimate_arena_bytes(
    nk_bytes: int,
    input_elements: int,
    output_elements: int,
    *,
    payload_bytes: int,
    network: str = "mlp",
    quantized: bool = False,
) -> tuple[int, int]:
    """Conservative fallback when ./netkit inspect is unavailable."""
    if quantized:
        # Flash-backed int8: loader metadata only (coefs stay in .nk blob).
        weight_guess = max(10240, input_elements * 4 + output_elements * 4)
    else:
        meta_guess = max(0, nk_bytes - payload_bytes)
        weight_guess = max(meta_guess, input_elements * 4 + output_elements * 4)
    after_load = weight_guess + 256
    io_scratch = (input_elements + output_elements) * 4 * 4
    after_forward = after_load + io_scratch

    if network == "cnn":
        if quantized:
            # CmsisQuantPlan: graph structs + largest int8 slot + CMSIS workspace.
            act_slot = input_elements * 28
            cnn_floor = weight_guess + act_slot + 48 * 1024 + 8192
        else:
            # CNN load allocates graph structs and ping-pong activations — often >> .nk file size.
            weight_bytes = max(0, nk_bytes - payload_bytes)
            cnn_floor = weight_bytes + nk_bytes * 4 + input_elements * 64
        after_load = max(after_load, cnn_floor)
        after_forward = max(after_forward, after_load)

    return after_load, after_forward


def _adjust_arena_for_flash(
    after_load: int,
    after_forward: int,
    payload_bytes: int,
    *,
    quantized: bool = False,
) -> tuple[int, int]:
    if payload_bytes <= 0 or quantized:
        return after_load, after_forward
    return max(0, after_load - payload_bytes), max(0, after_forward - payload_bytes)


def _probe_arena_bytes(nk_path: Path) -> tuple[int, int]:
    binary = _find_netkit_binary(nk_path)
    if binary is None:
        return (0, 0)
    proc = subprocess.run(
        [str(binary), "inspect", str(nk_path), "--full"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return (0, 0)
    text = proc.stdout
    load_match = re.search(r"after load:\s+(\d+) bytes", text)
    forward_match = re.search(r"after forward \(zero\):\s+(\d+) bytes", text)
    if not load_match or not forward_match:
        return (0, 0)
    return int(load_match.group(1)), int(forward_match.group(1))


def _cmsis_s8_workspace_slack(*, network: str, quantized: bool) -> int:
    """Host arena probe omits MCU CMSIS S8 kernel_workspace_ — add slack for flash builds."""
    if not quantized or network != "cnn":
        return 0
    return 48 * 1024


def _resolve_arena_bytes(
    nk_path: Path,
    nk_bytes: int,
    input_elements: int,
    output_elements: int,
    *,
    headroom_percent: int,
    payload_bytes: int,
    network: str = "mlp",
    quantized: bool = False,
) -> tuple[int, int, int]:
    after_load, after_forward = _probe_arena_bytes(nk_path)
    probed = after_forward > 0
    if not probed:
        after_load, after_forward = _estimate_arena_bytes(
            nk_bytes,
            input_elements,
            output_elements,
            payload_bytes=payload_bytes,
            network=network,
            quantized=quantized,
        )
        after_load, after_forward = _adjust_arena_for_flash(
            after_load,
            after_forward,
            payload_bytes,
            quantized=quantized,
        )
    workspace_slack = 0 if quantized else _cmsis_s8_workspace_slack(
        network=network, quantized=quantized
    )
    after_load += workspace_slack
    after_forward += workspace_slack
    recommended = _recommend_arena_bytes(after_forward, headroom_percent=headroom_percent)
    return after_load, after_forward, recommended


def compile_aot(
    nk_path: str | Path,
    output_dir: str | Path,
    *,
    language: Literal["cpp", "c"] | AotLanguage = AotLanguage.CPP,
    model_name: str | None = None,
    include_main: bool = False,
    optimize: bool = False,
    optimize_options: OptimizeOptions | None = None,
    arena_headroom_percent: int = 12,
    flash_section: bool = True,
    lower: bool = True,
    omit_final_softmax: bool = False,
) -> AotCompileResult:
    """Compile a .nk model into embeddable C or C++ source files.

    Default output is C++26 (.hpp + .cpp). Pass ``language="c"`` for C23 (.h + .c).
    Set ``optimize=True`` to apply stable graph optimizations (BN folding, linear dense
    merge) before embedding — fewer runtime layer dispatches, verified against the
    original model numerically.

    Emits measured arena sizing constants for MCU firmware (static buffer allocation).
    When ``./netkit inspect --full`` is available, arena bytes come from a probe load +
    zero-input forward (flash-backed weights). Otherwise a
    conservative estimate is used and ``weights_bytes + biases_bytes`` are subtracted
    when coefs stay in the embedded blob.

    C++ output is lowered by default (``lower=True``): static ``Kernels::`` call chain
    with embedded weight arrays, no ``.nk`` blob, loader, or ops resolver. Pass
    ``lower=False`` for the legacy embed-and-interpreter path. Lowering applies only
    to C++; C output always embeds the ``.nk`` blob.
    """
    path = Path(nk_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang = AotLanguage(language) if not isinstance(language, AotLanguage) else language
    stem = model_name or path.stem
    symbol = _sanitize_symbol(stem)

    raw_nk_bytes = path.read_bytes()
    header = unpack_header(raw_nk_bytes[:HEADER_BYTES])
    quantized = bool(header.get("flags", 0) & FLAG_HAS_QUANT)

    arch, weights = read_nk(path)
    tests = read_test_suite(path)
    optimizations_applied: tuple[str, ...] = ()
    if optimize:
        if quantized:
            raise ValueError("optimize is not supported for quantized .nk models")
        opt = optimize_nk(arch, weights, options=optimize_options)
        arch, weights = opt.arch, opt.weights
        optimizations_applied = tuple(opt.applied)

    if quantized:
        nk_bytes = raw_nk_bytes
        input_elements, output_elements = _io_elements_from_arch(arch)
    else:
        nk_bytes = arch_to_nk_bytes(arch, weights, tests=tests)
        header = unpack_header(nk_bytes[:HEADER_BYTES])
        input_elements, output_elements = _compute_io(arch, weights)
    payload_bytes = weight_payload_bytes(header)
    network = arch["network"]
    input_shape = arch["input"]
    use_lowered = (
        lang is AotLanguage.CPP and lower and not quantized and can_lower_arch(arch)
    )
    use_quant_lowered = (
        lang is AotLanguage.CPP and lower and quantized and can_lower_quantized_arch(arch)
    )

    if use_quant_lowered:
        quant_plan = plan_lowered_quant(path, omit_final_softmax=omit_final_softmax)
        after_load = quant_plan.arena_after_load
        after_forward = quant_plan.arena_after_forward
        arena_recommended = max(
            64,
            _recommend_arena_bytes(after_forward, headroom_percent=arena_headroom_percent),
        )
    elif use_lowered:
        plan = plan_lowered(
            arch,
            weights,
            omit_final_softmax=omit_final_softmax,
        )
        after_load = plan.arena_after_load
        after_forward = plan.arena_after_forward
        arena_recommended = max(
            64,
            _recommend_arena_bytes(after_forward, headroom_percent=arena_headroom_percent),
        )
    else:
        after_load, after_forward, arena_recommended = _resolve_arena_bytes(
            path,
            len(nk_bytes),
            input_elements,
            output_elements,
            headroom_percent=arena_headroom_percent,
            payload_bytes=payload_bytes,
            network=network,
            quantized=quantized,
        )
        arena_recommended = max(64, arena_recommended)

    blob_lines = _format_byte_array(nk_bytes)

    opt_comment = ""
    if optimizations_applied:
        joined = ", ".join(optimizations_applied)
        opt_comment = f"\n/* Optimizations: {joined} */"

    if lang is AotLanguage.CPP:
        header_path = out_dir / f"{symbol}_aot.hpp"
        source_path = out_dir / f"{symbol}_aot.cpp"
        if use_quant_lowered:
            header_path.write_text(
                render_lowered_quant_cpp_header(
                    symbol,
                    network,
                    input_elements,
                    output_elements,
                    input_shape,
                    quant_plan,
                    arena_recommended,
                ),
                encoding="utf-8",
            )
            source_path.write_text(
                opt_comment
                + render_lowered_quant_cpp_source(
                    symbol,
                    path,
                    quant_plan,
                    include_main,
                    flash_section,
                    omit_final_softmax=omit_final_softmax,
                ),
                encoding="utf-8",
            )
        elif use_lowered:
            header_path.write_text(
                render_lowered_cpp_header(
                    symbol,
                    network,
                    input_elements,
                    output_elements,
                    input_shape,
                    plan,
                    arena_recommended,
                ),
                encoding="utf-8",
            )
            source_path.write_text(
                opt_comment
                + render_lowered_cpp_source(
                    symbol,
                    plan,
                    include_main,
                    flash_section,
                ),
                encoding="utf-8",
            )
        else:
            header_path.write_text(
                _render_cpp_header(
                    symbol,
                    network,
                    input_elements,
                    output_elements,
                    input_shape,
                    after_load,
                    after_forward,
                    arena_recommended,
                    quantized=quantized,
                    quant_fast=False,
                ),
                encoding="utf-8",
            )
            source_path.write_text(
                opt_comment
                + _render_cpp_source(
                    symbol,
                    network,
                    input_elements,
                    output_elements,
                    input_shape,
                    len(nk_bytes),
                    blob_lines,
                    include_main,
                    flash_section,
                    quantized=quantized,
                    omit_final_softmax=omit_final_softmax,
                ),
                encoding="utf-8",
            )
    else:
        header_path = out_dir / f"{symbol}_aot.h"
        source_path = out_dir / f"{symbol}_aot.c"
        header_path.write_text(
            _render_c_header(
                symbol,
                input_elements,
                output_elements,
                after_load,
                after_forward,
                arena_recommended,
            ),
            encoding="utf-8",
        )
        source_path.write_text(
            opt_comment
            + _render_c_source(
                symbol,
                input_elements,
                output_elements,
                len(nk_bytes),
                blob_lines,
                include_main,
                flash_section,
            ),
            encoding="utf-8",
        )

    return AotCompileResult(
        model_name=symbol,
        language=lang,
        header_path=header_path,
        source_path=source_path,
        input_elements=input_elements,
        output_elements=output_elements,
        network=network,
        nk_bytes=len(nk_bytes),
        arena_bytes_after_load=after_load,
        arena_bytes_after_forward=after_forward,
        arena_bytes_recommended=arena_recommended,
        optimized=bool(optimizations_applied),
        optimizations_applied=optimizations_applied,
        lowered=use_lowered or use_quant_lowered,
        quant_fast=use_quant_lowered,
    )


def _render_cpp_header(
    symbol: str,
    network: str,
    input_elements: int,
    output_elements: int,
    input_shape: list[int],
    arena_after_load: int,
    arena_after_forward: int,
    arena_recommended: int,
    *,
    quantized: bool = False,
    quant_fast: bool = False,
) -> str:
    shape_literals = ", ".join(str(v) for v in input_shape)
    quant_fast_line = (
        "inline constexpr bool kQuantFastPath = true;\n" if quant_fast else ""
    )
    quant_lowered_line = "inline constexpr bool kQuantLowered = false;\ninline constexpr std::size_t kWorkspaceBytes = 0u;\n"
    if quantized:
        forward_decls = (
            "    bool forwardInt8(Arena& arena, const int8_t* input, int8_t* output) const;\n"
        )
    else:
        forward_decls = (
            "    bool forward(Arena& arena, const float* input, float* output) const;\n"
        )
    return f"""#pragma once
/* Generated by netkit AOT compiler — C++26 firmware-ready, links against libnetkit.a */

#include "arena.hpp"

#include <cstddef>
#include <cstdint>

namespace netkit::aot::{symbol} {{

inline constexpr const char* kName = "{symbol}";
inline constexpr const char* kNetwork = "{network}";
inline constexpr std::uint32_t kInputElements = {input_elements}u;
inline constexpr std::uint32_t kOutputElements = {output_elements}u;
inline constexpr std::uint32_t kInputShape[] = {{{shape_literals}}};
inline constexpr std::uint32_t kInputRank = {len(input_shape)}u;
inline constexpr std::size_t kArenaBytesAfterLoad = {arena_after_load}u;
inline constexpr std::size_t kArenaBytesAfterForward = {arena_after_forward}u;
inline constexpr std::size_t kArenaBytesRecommended = {arena_recommended}u;
{quant_lowered_line}{quant_fast_line}
extern const unsigned char kNkBlob[];
extern const std::size_t kNkBytes;

inline bool InitArena(Arena& arena, void* memory, std::size_t capacity)
{{
    if (!memory || capacity < kArenaBytesRecommended)
        return false;
    arena.init(memory, capacity);
    return true;
}}

class Model {{
public:
    Model() = default;

    bool load(Arena& arena);
{forward_decls}    [[nodiscard]] bool isLoaded() const {{ return loaded_; }}

private:
    bool loaded_ = false;
    void* network_ = nullptr;
}};

}}  // namespace netkit::aot::{symbol}
"""


def _render_cpp_source(
    symbol: str,
    network: str,
    input_elements: int,
    output_elements: int,
    input_shape: list[int],
    nk_size: int,
    blob_lines: str,
    include_main: bool,
    flash_section: bool,
    *,
    quantized: bool = False,
    omit_final_softmax: bool = False,
) -> str:
    omit_cpp = "true" if omit_final_softmax else "false"
    if network == "mlp":
        batch, features = input_shape
        load_body = f"""    std::array<std::uint32_t, kMaxTensorRank> shape{{}};
    std::uint32_t input_rank = 0;
    MLPNetwork* mlp = nullptr;
    const NkLoader::LoadResult result =
        NkLoader::LoadMLPFromBuffer(kNkBlob, kNkBytes, arena, mlp, shape, input_rank);
    if (result.status != NkLoader::LoadStatus::Ok || !mlp || !mlp->IsValid())
        return false;
    mlp->SetOmitFinalSoftmax({omit_cpp});
    network_ = mlp;"""
        forward_int8_body = f"""    auto* mlp = static_cast<MLPNetwork*>(network_);
    Tensor input_tensor = TensorFactory::View2DInt8(const_cast<int8_t*>(input), {batch}, {features});
    Tensor out_tensor = TensorFactory::View2DInt8(output, {batch}, kOutputElements / {batch});
    if (!input_tensor.data || !out_tensor.data)
        return false;
    mlp->forward(input_tensor, out_tensor, arena);
    return out_tensor.type == DataType::Int8;"""
        forward_float_body = f"""    auto* mlp = static_cast<MLPNetwork*>(network_);
    Tensor input_tensor = TensorFactory::Create2D(arena, {batch}, {features});
    if (!input_tensor.data)
        return false;
    float* input_data = static_cast<float*>(input_tensor.data);
    for (std::uint32_t i = 0; i < kInputElements; ++i)
        input_data[i] = input[i];
    Tensor out_tensor = TensorFactory::Create2D(arena, {batch}, kOutputElements / {batch});
    if (!out_tensor.data)
        return false;
    mlp->forward(input_tensor, out_tensor, arena);
    const float* out_data = static_cast<const float*>(out_tensor.data);
    for (std::uint32_t i = 0; i < kOutputElements; ++i)
        output[i] = out_data[i];
    return true;"""
    else:
        height, width, channels = input_shape
        load_body = f"""    std::array<std::uint32_t, kMaxTensorRank> shape{{}};
    std::uint32_t input_rank = 0;
    CNNNetwork* cnn = nullptr;
    const NkLoader::LoadResult result =
        NkLoader::LoadCNNFromBuffer(kNkBlob, kNkBytes, arena, cnn, shape, input_rank);
    if (result.status != NkLoader::LoadStatus::Ok || !cnn || !cnn->IsValid())
        return false;
    if (CmsisQuantPlan::Runtime* runtime = cnn->quant_runtime())
        runtime->omit_final_softmax = {omit_cpp};
    network_ = cnn;"""
        forward_int8_body = f"""    auto* cnn = static_cast<CNNNetwork*>(network_);
    CmsisQuantPlan::Runtime* runtime = cnn->quant_runtime();
    if (!runtime)
        return false;
    return CmsisQuantPlan::ForwardInt8ToBuffer(
        *runtime, *cnn, input, output, kOutputElements);"""
        forward_float_body = f"""    auto* cnn = static_cast<CNNNetwork*>(network_);
    Tensor input_tensor{{}};
    input_tensor.data = const_cast<float*>(input);
    input_tensor.type = DataType::Float32;
    input_tensor.rank = 3;
    input_tensor.shape[0] = {height};
    input_tensor.shape[1] = {width};
    input_tensor.shape[2] = {channels};
    input_tensor.stride[0] = {width} * {channels};
    input_tensor.stride[1] = {channels};
    input_tensor.stride[2] = 1;
    input_tensor.num_elements = kInputElements;
    input_tensor.bytes = input_tensor.num_elements * sizeof(float);
    Tensor& out_tensor = cnn->forward(input_tensor, arena);
    if (!out_tensor.data || out_tensor.type != DataType::Float32)
        return false;
    const float* out_data = static_cast<const float*>(out_tensor.data);
    for (std::uint32_t i = 0; i < kOutputElements; ++i)
        output[i] = out_data[i];
    return true;"""

    main_block = ""
    if include_main:
        if quantized:
            main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include "arena_util.hpp"
#include <cstdio>
#include <stdalign.h>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[netkit::aot::{symbol}::kArenaBytesRecommended];
    Arena arena;
    if (!netkit::aot::{symbol}::InitArena(arena, arena_mem, sizeof(arena_mem)))
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena))
        return 1;

    int8_t input[netkit::aot::{symbol}::kInputElements] = {{0}};
    int8_t output[netkit::aot::{symbol}::kOutputElements] = {{0}};
    if (!model.forwardInt8(arena, input, output))
        return 1;

    for (std::uint32_t i = 0; i < netkit::aot::{symbol}::kOutputElements; ++i)
        std::printf(i ? ",%d" : "%d", static_cast<int>(output[i]));
    std::printf("\\n");
    return 0;
}}
#endif
"""
        else:
            main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include "arena_util.hpp"
#include <cstdio>
#include <stdalign.h>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[netkit::aot::{symbol}::kArenaBytesRecommended];
    Arena arena;
    if (!netkit::aot::{symbol}::InitArena(arena, arena_mem, sizeof(arena_mem)))
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena))
        return 1;

    float input[netkit::aot::{symbol}::kInputElements] = {{0.0f}};
    float output[netkit::aot::{symbol}::kOutputElements] = {{0.0f}};
    if (!model.forward(arena, input, output))
        return 1;

    for (std::uint32_t i = 0; i < netkit::aot::{symbol}::kOutputElements; ++i)
        std::printf(i ? ",%.6f" : "%.6f", output[i]);
    std::printf("\\n");
    return 0;
}}
#endif
"""

    flash_attr = ""
    if flash_section:
        flash_attr = (
            "\n#if defined(__GNUC__)\n"
            "#if defined(__ELF__)\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((section(\".rodata\"), aligned(8)))\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((aligned(8)))\n"
            "#endif\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST\n"
            "#endif\n"
        )
    else:
        flash_attr = "\n#define NETKIT_AOT_FLASH_CONST\n"

    if quantized:
        forward_method = f"""
bool Model::forwardInt8(Arena& arena, const int8_t* input, int8_t* output) const
{{
    if (!loaded_ || !network_ || !input || !output)
        return false;
{forward_int8_body}
}}
"""
    else:
        forward_method = f"""
bool Model::forward(Arena& arena, const float* input, float* output) const
{{
    if (!loaded_ || !network_ || !input || !output)
        return false;
{forward_float_body}
}}
"""

    return f"""/* Generated by netkit AOT compiler — compile with -std=c++26 for firmware */
{flash_attr}
#include "{symbol}_aot.hpp"

#include "arena.hpp"
#include "cnn.hpp"
#include "cmsis_quant_plan.hpp"
#include "mlp.hpp"
#include "nk_loader.hpp"
#include "quant_output.hpp"
#include "tensor_factory.hpp"

#include <array>

namespace netkit::aot::{symbol} {{

const std::size_t kNkBytes = {nk_size};

NETKIT_AOT_FLASH_CONST const unsigned char kNkBlob[kNkBytes] = {{
{blob_lines}
}};

bool Model::load(Arena& arena)
{{
{load_body}
    loaded_ = true;
    return true;
}}
{forward_method}
}}  // namespace netkit::aot::{symbol}
{main_block}
"""


def _render_c_header(
    symbol: str,
    input_elements: int,
    output_elements: int,
    arena_after_load: int,
    arena_after_forward: int,
    arena_recommended: int,
) -> str:
    guard = f"NETKIT_{symbol.upper()}_AOT_H"
    prefix = symbol.upper()
    return f"""#ifndef {guard}
#define {guard}
/* Generated by netkit AOT compiler — C23 firmware-ready, links against libnetkit.a */

#include <stddef.h>
#include <stdint.h>

#include "netkit.h"

#define {prefix}_AOT_INPUT_ELEMENTS {input_elements}u
#define {prefix}_AOT_OUTPUT_ELEMENTS {output_elements}u
#define {prefix}_AOT_ARENA_BYTES_AFTER_LOAD {arena_after_load}u
#define {prefix}_AOT_ARENA_BYTES_AFTER_FORWARD {arena_after_forward}u
#define {prefix}_AOT_ARENA_BYTES_RECOMMENDED {arena_recommended}u

extern const unsigned char {symbol}_aot_nk[];
extern const size_t {symbol}_aot_nk_size;

static inline nk_status_t {symbol}_aot_init_arena(nk_arena_t* arena, void* memory, size_t capacity)
{{
    if (!arena || !memory || capacity < {prefix}_AOT_ARENA_BYTES_RECOMMENDED)
        return NK_ERR_INVALID_ARGUMENT;
    nk_arena_init(arena, memory, capacity);
    return NK_OK;
}}

nk_status_t {symbol}_aot_load(nk_arena_t* arena, nk_model_t* model);
nk_status_t {symbol}_aot_run(const nk_model_t* model,
                             nk_arena_t* arena,
                             const float* input,
                             float* output,
                             uint32_t* output_count);

#endif /* {guard} */
"""


def _render_c_source(
    symbol: str,
    input_elements: int,
    output_elements: int,
    nk_size: int,
    blob_lines: str,
    include_main: bool,
    flash_section: bool,
) -> str:
    main_block = ""
    if include_main:
        main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include <stdio.h>
#include <stdalign.h>

int main(void)
{{
    alignas(max_align_t) static unsigned char arena_mem[{symbol.upper()}_AOT_ARENA_BYTES_RECOMMENDED];
    nk_arena_t arena;
    if ({symbol}_aot_init_arena(&arena, arena_mem, sizeof(arena_mem)) != NK_OK)
        return 1;

    nk_model_t model;
    if ({symbol}_aot_load(&arena, &model) != NK_OK)
        return 1;

    float input[{symbol.upper()}_AOT_INPUT_ELEMENTS] = {{0.0f}};
    float output[{symbol.upper()}_AOT_OUTPUT_ELEMENTS] = {{0.0f}};
    uint32_t output_count = 0;
    if ({symbol}_aot_run(&model, &arena, input, output, &output_count) != NK_OK)
        return 1;

    for (uint32_t i = 0; i < output_count; ++i)
        printf(i ? ",%.6f" : "%.6f", output[i]);
    printf("\\n");
    return 0;
}}
#endif
"""

    flash_attr = ""
    if flash_section:
        flash_attr = (
            "\n#if defined(__GNUC__)\n"
            "#if defined(__ELF__)\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((section(\".rodata\"), aligned(8)))\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST __attribute__((aligned(8)))\n"
            "#endif\n"
            "#else\n"
            "#define NETKIT_AOT_FLASH_CONST\n"
            "#endif\n"
        )
    else:
        flash_attr = "\n#define NETKIT_AOT_FLASH_CONST\n"

    return f"""/* Generated by netkit AOT compiler — compile with -std=c23 for firmware */
{flash_attr}
#include "{symbol}_aot.h"

NETKIT_AOT_FLASH_CONST const unsigned char {symbol}_aot_nk[] = {{
{blob_lines}
}};

const size_t {symbol}_aot_nk_size = sizeof({symbol}_aot_nk);

nk_status_t {symbol}_aot_load(nk_arena_t* arena, nk_model_t* model)
{{
    if (!arena || !model)
        return NK_ERR_INVALID_ARGUMENT;
    return nk_model_load_memory({symbol}_aot_nk, {symbol}_aot_nk_size, arena, model);
}}

nk_status_t {symbol}_aot_run(const nk_model_t* model,
                             nk_arena_t* arena,
                             const float* input,
                             float* output,
                             uint32_t* output_count)
{{
    if (!model || !arena || !input || !output || !output_count)
        return NK_ERR_INVALID_ARGUMENT;
    return nk_model_run(model,
                        arena,
                        input,
                        {symbol.upper()}_AOT_INPUT_ELEMENTS,
                        output,
                        {symbol.upper()}_AOT_OUTPUT_ELEMENTS,
                        output_count);
}}
{main_block}
"""
