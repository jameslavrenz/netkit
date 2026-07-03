#!/usr/bin/env bash
# Build and run MNIST kernel benchmarks (reference vs CMSIS-DSP on CPU desktop).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BENCH_SRC="benchmarks/mnist_kernel_bench.cpp"
BENCH_REF="benchmarks/mnist_kernel_bench_ref"
BENCH_DSP="benchmarks/mnist_kernel_bench_dsp"
BENCH_REF_OBJ="benchmarks/mnist_kernel_bench_ref.o"
BENCH_DSP_OBJ="benchmarks/mnist_kernel_bench_dsp.o"

BENCH_CXXFLAGS="-O2 -std=c++26 -Wall -Wextra -Iinclude"
BENCH_CFLAGS="-O2 -std=c23 -Wall -Wextra -Iinclude"

if [[ ! -f "third_party/CMSIS-DSP/Include/arm_math.h" ]]; then
  ./tools/fetch_cmsis_dsp.sh
fi

if [[ ! -f "models/mnist_mlp.nk" || ! -f "models/mnist_cnn.nk" ]]; then
  echo "Missing MNIST models — run: make export-mnist-all" >&2
  exit 1
fi

clean_runtime_objects() {
  rm -f src/*.o src/layer_ops/*.o libnetkit.a build/cmsis_dsp/*.o 2>/dev/null || true
}

echo "=== Building reference kernel library (-O2) ==="
clean_runtime_objects
make NETKIT_CMSIS_DSP=0 lib \
  CXXFLAGS="-fcolor-diagnostics -fansi-escape-codes -g ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -D__GNUC_PYTHON__" \
  CFLAGS="-fcolor-diagnostics -fansi-escape-codes -g ${BENCH_CFLAGS} -DNETKIT_TARGET_CPU=1 -D__GNUC_PYTHON__"

clang++ ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -c "${BENCH_SRC}" -o "${BENCH_REF_OBJ}"
clang++ ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -o "${BENCH_REF}" "${BENCH_REF_OBJ}" libnetkit.a

echo "=== Building CMSIS-DSP kernel library (-O2) ==="
clean_runtime_objects
make NETKIT_CMSIS_DSP=1 lib \
  CXXFLAGS="-fcolor-diagnostics -fansi-escape-codes -g ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -D__GNUC_PYTHON__ -DARM_MATH_LOOPUNROLL" \
  CFLAGS="-fcolor-diagnostics -fansi-escape-codes -g ${BENCH_CFLAGS} -DNETKIT_TARGET_CPU=1 -D__GNUC_PYTHON__ -DARM_MATH_LOOPUNROLL"

clang++ ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -DNETKIT_USE_CMSIS_DSP=1 -c "${BENCH_SRC}" -o "${BENCH_DSP_OBJ}"
clang++ ${BENCH_CXXFLAGS} -DNETKIT_TARGET_CPU=1 -DNETKIT_USE_CMSIS_DSP=1 -o "${BENCH_DSP}" "${BENCH_DSP_OBJ}" libnetkit.a

echo
echo "=== MNIST inference benchmark (CPU desktop, NETKIT_TARGET=cpu) ==="
echo "CMSIS-DSP accelerates vector/matrix ops only; conv/pool stay on reference kernels."
echo

REF_OUT="$("${BENCH_REF}")"
DSP_OUT="$("${BENCH_DSP}")"

printf '%s\n' "${REF_OUT}"
printf '%s\n' "${DSP_OUT}"

python3 - "${REF_OUT}" "${DSP_OUT}" <<'PY'
import re
import sys

def parse_block(text):
    rows = {}
    for line in text.strip().splitlines():
        m = re.search(r"model=(\S+).*backend=(\S+).*avg_ms=([0-9.]+)", line)
        if m:
            rows[m.group(1)] = float(m.group(3))
    return rows

ref = parse_block(sys.argv[1])
dsp = parse_block(sys.argv[2])

print()
print("=== Speedup (reference / cmsis-dsp) ===")
for model in ("mnist_mlp", "mnist_cnn"):
    if model not in ref or model not in dsp:
        print(f"{model}: missing results")
        continue
    ratio = ref[model] / dsp[model] if dsp[model] > 0 else float("inf")
    pct = (ratio - 1.0) * 100.0
    print(
        f"{model}: reference={ref[model]:.4f} ms  cmsis-dsp={dsp[model]:.4f} ms  "
        f"speedup={ratio:.2f}x ({pct:+.1f}%)"
    )
PY
