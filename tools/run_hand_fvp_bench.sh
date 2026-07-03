#!/usr/bin/env bash
# Build Cortex-M4F hand-model firmware (reference vs CMSIS-NN+CMSIS-DSP) and run under Arm FVP.
# One model per ELF / FVP run. Native Linux FVP is preferred; Docker is macOS fallback only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FVP_DIR="${NETKIT_FVP_DIR:-benchmarks/fvp}"
# Use NETKIT_FVP_HAND_TIMELIMIT (not NETKIT_FVP_TIMELIMIT) so a stray local export cannot
# override the per-run wall-clock cap (e.g. NETKIT_FVP_TIMELIMIT=900 on a dev machine).
FVP_TIMELIMIT="${NETKIT_FVP_HAND_TIMELIMIT:-60}"

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 not found — $2" >&2
    exit 1
  fi
}

setup_arm_toolchain_path() {
  if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
      return 0
    fi
  fi
  local candidate
  for candidate in \
    "${HOME}/arm-gnu-toolchain/bin" \
    ; do
    if [[ -d "${candidate}" && -x "${candidate}/arm-none-eabi-gcc" ]]; then
      export PATH="${candidate}:${PATH}"
      if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
        return 0
      fi
    fi
  done
  if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    for candidate in "${GITHUB_WORKSPACE}"/arm-gnu-toolchain-*/bin; do
      if [[ -d "${candidate}" && -x "${candidate}/arm-none-eabi-gcc" ]]; then
        export PATH="${candidate}:${PATH}"
        if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
          return 0
        fi
      fi
    done
  fi
  for candidate in \
    /Applications/ArmGNUToolchain/*/arm-none-eabi/bin \
    ; do
    if [[ -x "${candidate}/arm-none-eabi-gcc" ]]; then
      export PATH="${candidate}:${PATH}"
      if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
        return 0
      fi
    fi
  done
  return 1
}

FVP_CONFIG="${ROOT}/${FVP_DIR}/fvp_m4_config.txt"
FVP_VERSION="${NETKIT_FVP_VERSION:-11.31.28}"
FVP_MODEL="${NETKIT_FVP_MODEL:-FVP_MPS2_Cortex-M4}"

fvp_docker_ready() {
  command -v docker >/dev/null 2>&1 && docker image inspect "fvp:${FVP_VERSION}" >/dev/null 2>&1
}

find_fvp() {
  # Native FVP (Linux/Windows) — preferred on CI and Linux desktops.
  local candidate
  for candidate in FVP_MPS2_Cortex-M4 FVP_Cortex-M4; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "native:${candidate}"
      return 0
    fi
  done
  if [[ -n "${NETKIT_FVP:-}" && -x "${NETKIT_FVP}" ]]; then
    echo "${NETKIT_FVP}"
    return 0
  fi
  # Docker fallback for macOS only (unless explicitly skipped for CI).
  if [[ "${NETKIT_FVP_SKIP_DOCKER:-0}" != "1" ]] && [[ "$(uname -s)" == "Darwin" ]] && fvp_docker_ready; then
    echo "docker:fvp:${FVP_VERSION}"
    return 0
  fi
  return 1
}

print_license_help() {
  echo "" >&2
  echo "Arm FVP requires an activated user-based license in ~/.armlm." >&2
  echo "On GitHub Actions, the fvp-bench workflow uses ARM-software/cmsis-actions/armlm@v1." >&2
  echo "Locally (macOS Docker), activate the free MDK Community license (one-time):" >&2
  echo "  mkdir -p ~/.armlm" >&2
  echo "  docker run --rm -it -u root \\" >&2
  echo "    --mount type=bind,src=\${HOME}/.armlm,dst=/root/.armlm \\" >&2
  echo "    -e ARMLM_CACHED_LICENSES_LOCATION=/root/.armlm \\" >&2
  echo "    fvp:${FVP_VERSION} /opt/avh-fvp/arm_license_management_utilities/armlm activate \\" >&2
  echo "    --server https://mdk-preview.keil.arm.com --product KEMDK-COM0" >&2
}

run_fvp_docker() {
  local elf="$1"
  local abs_elf abs_dir elf_base abs_config
  abs_elf="$(cd "$(dirname "$elf")" && pwd)/$(basename "$elf")"
  abs_dir="$(dirname "$abs_elf")"
  elf_base="$(basename "$abs_elf")"
  abs_config="$(cd "$(dirname "$FVP_CONFIG")" && pwd)/$(basename "$FVP_CONFIG")"

  docker run --rm \
    --mount "type=bind,src=${HOME}/.armlm,dst=/root/.armlm" \
    --mount "type=bind,src=${HOME}/,dst=${HOME}/" \
    --workdir "${abs_dir}" \
    -e ARMLM_CACHED_LICENSES_LOCATION=/root/.armlm \
    -u root \
    "fvp:${FVP_VERSION}" "${FVP_MODEL}" \
    -f "${abs_config}" \
    -a "${elf_base}" --stat -q --timelimit "${FVP_TIMELIMIT}"
}

run_fvp_native() {
  local fvp_bin="$1"
  local elf="$2"
  local abs_elf
  abs_elf="$(cd "$(dirname "$elf")" && pwd)/$(basename "$elf")"

  "${fvp_bin}" -f "${FVP_CONFIG}" -a "${abs_elf}" --stat -q --timelimit "${FVP_TIMELIMIT}" 2>&1
}

run_fvp() {
  local fvp="$1"
  local elf="$2"
  local out fvp_bin

  if [[ "${fvp}" == docker:fvp:* ]]; then
    out="$(run_fvp_docker "${elf}" 2>&1)" || {
      if rg -q 'license error|No active licenses|assigned to another user' <<<"${out}"; then
        echo "${out}" >&2
        print_license_help
        exit 1
      fi
      echo "${out}" >&2
      return 1
    }
    printf '%s\n' "${out}"
    return 0
  fi

  if [[ "${fvp}" == native:* ]]; then
    fvp_bin="${fvp#native:}"
  else
    fvp_bin="${fvp}"
  fi

  if ! out="$(run_fvp_native "${fvp_bin}" "${elf}")"; then
    if rg -q 'license error|No active licenses|assigned to another user' <<<"${out}"; then
      echo "${out}" >&2
      print_license_help
      exit 1
    fi
    echo "${out}" >&2
    return 1
  fi
  printf '%s\n' "${out}"
}

setup_arm_toolchain_path || true
require_tool arm-none-eabi-gcc "install Arm GNU Toolchain (arm-none-eabi) — see docs/TESTING.md"

for model in models/mlp_hand.nk models/cnn_hand.nk; do
  if [[ ! -f "${model}" ]]; then
    echo "Missing ${model} — run: make export-nk" >&2
    exit 1
  fi
done

make cmsis-init

PYTHONPATH=python python3 tools/embed_hand_fvp_data.py

echo "=== Building hand FVP firmware (reference + CMSIS, mlp + cnn, 64 KiB arena) ==="
make -C benchmarks/fvp all-models

FVP_BIN=""
if FVP_BIN="$(find_fvp)"; then
  echo "=== Running under FVP: ${FVP_BIN} ==="
  echo "models=mlp_hand.nk cnn_hand.nk arena=64KiB cycle_target=100 (one model per run)"
  echo "target=Cortex-M4F (-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard)"
  echo "timelimit=${FVP_TIMELIMIT}s per run (NETKIT_FVP_HAND_TIMELIMIT)"
  echo

  REF_OUT=""
  CMSIS_OUT=""
  for model in mlp cnn; do
    ref_elf="${FVP_DIR}/hand_fvp_bench_ref_${model}.elf"
    cmsis_elf="${FVP_DIR}/hand_fvp_bench_cmsis_${model}.elf"
    echo "--- FVP run: reference / ${model}_hand ---"
    REF_OUT+="$(run_fvp "${FVP_BIN}" "${ref_elf}")"
    REF_OUT+=$'\n'
    echo "--- FVP run: cmsis / ${model}_hand ---"
    CMSIS_OUT+="$(run_fvp "${FVP_BIN}" "${cmsis_elf}")"
    CMSIS_OUT+=$'\n'
  done

  printf '%s\n' "${REF_OUT}"
  echo "---"
  printf '%s\n' "${CMSIS_OUT}"

  python3 - "${REF_OUT}" "${CMSIS_OUT}" <<'PY'
import re
import sys

def parse(text):
    rows = {}
    for line in text.strip().splitlines():
        m = re.search(
            r"model=(\S+).*backend=(\S+).*avg_cycles=(\d+)",
            line,
        )
        if m:
            rows[m.group(1)] = int(m.group(3))
    return rows

ref = parse(sys.argv[1])
cmsis = parse(sys.argv[2])

print()
print("=== Speedup (reference cycles / cmsis cycles) ===")
for model in ("mlp_hand", "cnn_hand"):
    if model not in ref or model not in cmsis:
        print(f"{model}: missing results")
        continue
    ratio = ref[model] / cmsis[model] if cmsis[model] else float("inf")
    pct = (ratio - 1.0) * 100.0
    print(
        f"{model}: reference={ref[model]} cycles  cmsis={cmsis[model]} cycles  "
        f"speedup={ratio:.2f}x ({pct:+.1f}%)"
    )

missing = [m for m in ("mlp_hand", "cnn_hand") if m not in ref or m not in cmsis]
if missing:
    sys.exit(1)
PY
else
  echo ""
  echo "Arm FVP not found — firmware built successfully under benchmarks/fvp/:"
  echo "  hand_fvp_bench_ref_{mlp,cnn}.elf"
  echo "  hand_fvp_bench_cmsis_{mlp,cnn}.elf"
  echo ""
  echo "On Linux CI, use the fvp-bench GitHub Actions workflow."
  echo "On macOS, install FVPs-on-Mac Docker wrappers or set NETKIT_FVP."
  exit 0
fi
