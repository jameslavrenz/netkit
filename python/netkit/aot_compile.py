"""Ahead-of-time compiler: embed a .nk model as C or C++ source for the netkit runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np

from .format import HEADER_BYTES, unpack_header
from .reader import read_nk
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


def _read_model_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"truncated .nk file: {path}")
    header = unpack_header(raw[:HEADER_BYTES])
    network = header["network_kind"].name.lower()
    if network not in {"mlp", "cnn"}:
        raise ValueError(f"unsupported network kind in {path}: {network}")
    return raw


def compile_aot(
    nk_path: str | Path,
    output_dir: str | Path,
    *,
    language: Literal["cpp", "c"] | AotLanguage = AotLanguage.CPP,
    model_name: str | None = None,
    include_main: bool = False,
) -> AotCompileResult:
    """Compile a .nk model into embeddable C or C++ source files.

    Default output is C++26 (.hpp + .cpp). Pass ``language="c"`` for C23 (.h + .c).
    """
    path = Path(nk_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang = AotLanguage(language) if not isinstance(language, AotLanguage) else language
    stem = model_name or path.stem
    symbol = _sanitize_symbol(stem)

    nk_bytes = _read_model_bytes(path)
    arch, weights = read_nk(path)
    input_elements, output_elements = _compute_io(arch, weights)
    network = arch["network"]
    input_shape = arch["input"]

    blob_lines = _format_byte_array(nk_bytes)

    if lang is AotLanguage.CPP:
        header_path = out_dir / f"{symbol}_aot.hpp"
        source_path = out_dir / f"{symbol}_aot.cpp"
        header_path.write_text(
            _render_cpp_header(symbol, network, input_elements, output_elements, input_shape),
            encoding="utf-8",
        )
        source_path.write_text(
            _render_cpp_source(
                symbol,
                network,
                input_elements,
                output_elements,
                input_shape,
                len(nk_bytes),
                blob_lines,
                include_main,
            ),
            encoding="utf-8",
        )
    else:
        header_path = out_dir / f"{symbol}_aot.h"
        source_path = out_dir / f"{symbol}_aot.c"
        header_path.write_text(
            _render_c_header(symbol, input_elements, output_elements),
            encoding="utf-8",
        )
        source_path.write_text(
            _render_c_source(
                symbol,
                input_elements,
                output_elements,
                len(nk_bytes),
                blob_lines,
                include_main,
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
    )


def _render_cpp_header(
    symbol: str, network: str, input_elements: int, output_elements: int, input_shape: list[int]
) -> str:
    shape_literals = ", ".join(str(v) for v in input_shape)
    return f"""#pragma once
/* Generated by netkit AOT compiler — C++26, links against libnetkit.a */

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

extern const unsigned char kNkBlob[];
extern const std::size_t kNkBytes;

class Model {{
public:
    Model() = default;

    bool load(Arena& arena);
    bool forward(Arena& arena, const float* input, float* output) const;
    [[nodiscard]] bool isLoaded() const {{ return loaded_; }}

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
) -> str:
    if network == "mlp":
        batch, features = input_shape
        load_body = """    std::array<std::uint32_t, kMaxTensorRank> shape{};
    std::uint32_t input_rank = 0;
    MLPNetwork* mlp = nullptr;
    const NkLoader::LoadResult result =
        NkLoader::LoadMLPFromBuffer(kNkBlob, kNkBytes, arena, mlp, shape, input_rank);
    if (result.status != NkLoader::LoadStatus::Ok || !mlp || !mlp->IsValid())
        return false;
    network_ = mlp;"""
        forward_body = f"""    auto* mlp = static_cast<MLPNetwork*>(network_);
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
        load_body = """    std::array<std::uint32_t, kMaxTensorRank> shape{};
    std::uint32_t input_rank = 0;
    CNNNetwork* cnn = nullptr;
    const NkLoader::LoadResult result =
        NkLoader::LoadCNNFromBuffer(kNkBlob, kNkBytes, arena, cnn, shape, input_rank);
    if (result.status != NkLoader::LoadStatus::Ok || !cnn || !cnn->IsValid())
        return false;
    network_ = cnn;"""
        forward_body = f"""    auto* cnn = static_cast<CNNNetwork*>(network_);
    float input_buffer[kInputElements] = {{}};
    for (std::uint32_t i = 0; i < kInputElements; ++i)
        input_buffer[i] = input[i];
    Tensor input_tensor{{}};
    input_tensor.data = input_buffer;
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
    if (!out_tensor.data)
        return false;
    const float* out_data = static_cast<const float*>(out_tensor.data);
    for (std::uint32_t i = 0; i < kOutputElements; ++i)
        output[i] = out_data[i];
    return true;"""

    main_block = ""
    if include_main:
        main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include "arena_util.hpp"
#include <cstdio>

int main(void)
{{
    alignas(std::max_align_t) static unsigned char arena_mem[Arena::kDefaultCapacity];
    ArenaUtil::Scoped arena_scope(Arena::kDefaultCapacity, arena_mem);
    if (!arena_scope)
        return 1;

    netkit::aot::{symbol}::Model model;
    if (!model.load(arena_scope.Get()))
        return 1;

    float input[netkit::aot::{symbol}::kInputElements] = {{0.0f}};
    float output[netkit::aot::{symbol}::kOutputElements] = {{0.0f}};
    if (!model.forward(arena_scope.Get(), input, output))
        return 1;

    for (std::uint32_t i = 0; i < netkit::aot::{symbol}::kOutputElements; ++i)
        std::printf(i ? ",%.6f" : "%.6f", output[i]);
    std::printf("\\n");
    return 0;
}}
#endif
"""

    return f"""/* Generated by netkit AOT compiler — compile with -std=c++26 */

#include "{symbol}_aot.hpp"

#include "arena.hpp"
#include "cnn.hpp"
#include "mlp.hpp"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

#include <array>

namespace netkit::aot::{symbol} {{

const std::size_t kNkBytes = {nk_size};

const unsigned char kNkBlob[kNkBytes] = {{
{blob_lines}
}};

bool Model::load(Arena& arena)
{{
{load_body}
    loaded_ = true;
    return true;
}}

bool Model::forward(Arena& arena, const float* input, float* output) const
{{
    if (!loaded_ || !network_ || !input || !output)
        return false;
{forward_body}
}}

}}  // namespace netkit::aot::{symbol}
{main_block}
"""


def _render_c_header(symbol: str, input_elements: int, output_elements: int) -> str:
    guard = f"NETKIT_{symbol.upper()}_AOT_H"
    prefix = symbol.upper()
    return f"""#ifndef {guard}
#define {guard}
/* Generated by netkit AOT compiler — C23, links against libnetkit.a */

#include <stddef.h>
#include <stdint.h>

#include "netkit.h"

#define {prefix}_AOT_INPUT_ELEMENTS {input_elements}u
#define {prefix}_AOT_OUTPUT_ELEMENTS {output_elements}u

extern const unsigned char {symbol}_aot_nk[];
extern const size_t {symbol}_aot_nk_size;

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
) -> str:
    main_block = ""
    if include_main:
        main_block = f"""

#ifdef NETKIT_AOT_MAIN
#include <stdio.h>
#include <stdalign.h>

int main(void)
{{
    alignas(max_align_t) static unsigned char arena_mem[NK_ARENA_DEFAULT_CAPACITY];
    nk_arena_t arena;
#if defined(NETKIT_ARENA_HEAP)
    if (nk_arena_init_heap(&arena, NK_ARENA_DEFAULT_CAPACITY) != NK_OK)
        return 1;
#else
    nk_arena_init(&arena, arena_mem, sizeof(arena_mem));
#endif

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

#if defined(NETKIT_ARENA_HEAP)
    nk_arena_destroy_heap(&arena);
#endif
    return 0;
}}
#endif
"""

    return f"""/* Generated by netkit AOT compiler — compile with -std=c23 */

#include "{symbol}_aot.h"

const unsigned char {symbol}_aot_nk[] = {{
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
