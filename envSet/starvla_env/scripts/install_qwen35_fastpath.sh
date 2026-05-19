#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="${1:-$(starvla_default_env_dir "${REPO_ROOT}")}"
PYTHON="${ENV_DIR}/bin/python"
SRC_DIR="${ENVSET_ROOT}/sources/qwen35_fastpath"
LOG_DIR="${ENVSET_ROOT}/logs/qwen35_fastpath"

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing python: ${PYTHON}" >&2
  exit 1
fi
if [ ! -d "${SRC_DIR}" ]; then
  echo "Missing Qwen3.5 fast-path source dir: ${SRC_DIR}" >&2
  exit 1
fi

causal_src="$(find "${SRC_DIR}" -maxdepth 1 -name 'causal_conv1d-*.tar.gz' | sort | tail -n 1)"
fla_src="$(find "${SRC_DIR}" -maxdepth 1 -name 'flash_linear_attention-*.tar.gz' | sort | tail -n 1)"
fla_core_pkg="$(find "${SRC_DIR}" -maxdepth 1 \( -name 'fla_core-*.whl' -o -name 'fla-core-*.whl' -o -name 'fla_core-*.tar.gz' -o -name 'fla-core-*.tar.gz' \) | sort | tail -n 1)"
if [ -z "${causal_src}" ] || [ -z "${fla_src}" ]; then
  echo "Missing causal-conv1d or flash-linear-attention source packages under ${SRC_DIR}" >&2
  exit 1
fi
if [ -z "${fla_core_pkg}" ]; then
  echo "[starvla-qwen35-fastpath] missing fla-core package under ${SRC_DIR}; FLA CUDA fast-path cannot be completed." >&2
fi

causal_conv1d_ok() {
  "${PYTHON}" - <<'PY' >/dev/null 2>&1
import causal_conv1d
PY
}

fla_runtime_ok() {
  "${PYTHON}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as metadata
from pathlib import Path

metadata.version("fla-core")
import fla

roots = [Path(path) for path in getattr(fla, "__path__", [])]
if not any((root / "modules").is_dir() for root in roots):
    raise SystemExit("missing fla.modules")
if not any((root / "ops" / "gated_delta_rule").is_dir() for root in roots):
    raise SystemExit("missing fla.ops.gated_delta_rule")

try:
    import torch
    cuda_available = torch.cuda.is_available()
except Exception:
    cuda_available = False

if cuda_available:
    from fla.modules import FusedRMSNormGated  # noqa: F401
    from fla.ops.gated_delta_rule import (  # noqa: F401
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
PY
}

transformers_runtime_ok() {
  "${PYTHON}" - <<'PY' >/dev/null 2>&1
import transformers

for attr in ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    getattr(transformers, attr)
PY
}

show_transformers_runtime_error() {
  "${PYTHON}" - <<'PY' 2>&1 || true
import traceback

try:
    import transformers
    print("transformers", getattr(transformers, "__version__", "unknown"))
    for attr in ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
        try:
            getattr(transformers, attr)
            print(f"{attr}: OK")
        except Exception:
            print(f"{attr}: FAIL")
            traceback.print_exc()
except Exception:
    print("import transformers: FAIL")
    traceback.print_exc()
PY
}

repair_transformers_runtime() {
  echo "[starvla-qwen35-fastpath] refreshing Transformers overlay for Qwen3.5 runtime compatibility."
  "${SCRIPT_DIR}/apply_transformers_5_2_overlay.sh" "${ENV_DIR}"
}

ensure_transformers_runtime() {
  if transformers_runtime_ok; then
    echo "[starvla-qwen35-fastpath] Transformers Qwen3.5/Qwen3-VL runtime OK."
    return 0
  fi
  echo "[starvla-qwen35-fastpath] Transformers runtime check failed before repair:" >&2
  show_transformers_runtime_error >&2
  repair_transformers_runtime
  if transformers_runtime_ok; then
    echo "[starvla-qwen35-fastpath] Transformers runtime repaired without rebuilding CUDA extensions."
    return 0
  fi
  echo "[starvla-qwen35-fastpath] Transformers runtime still failed after overlay repair:" >&2
  show_transformers_runtime_error >&2
  return 1
}

install_fla_stack() {
  if [ -z "${fla_core_pkg}" ]; then
    echo "[starvla-qwen35-fastpath] skipped FLA core install because no offline fla-core package is available." >&2
    return 1
  fi
  run_logged fla_core_install "${PYTHON}" -m pip install --no-index --no-deps --force-reinstall "${fla_core_pkg}"
  run_logged flash_linear_attention_install "${PYTHON}" -m pip install --no-index --no-deps --force-reinstall --no-build-isolation --no-cache-dir "${fla_src}"
}

if causal_conv1d_ok && fla_runtime_ok; then
  echo "[starvla-qwen35-fastpath] Qwen3.5 fast-path packages already import OK; skipped CUDA rebuild."
  ensure_transformers_runtime
  exit 0
fi

find_cxx() {
  if [ -n "${CXX:-}" ] && command -v "${CXX%% *}" >/dev/null 2>&1; then
    command -v "${CXX%% *}"
    return 0
  fi
  if command -v g++ >/dev/null 2>&1; then
    command -v g++
    return 0
  fi
  if command -v c++ >/dev/null 2>&1; then
    command -v c++
    return 0
  fi
  return 1
}

find_cc() {
  if [ -n "${CC:-}" ] && command -v "${CC%% *}" >/dev/null 2>&1; then
    command -v "${CC%% *}"
    return 0
  fi
  if command -v gcc >/dev/null 2>&1; then
    command -v gcc
    return 0
  fi
  if command -v cc >/dev/null 2>&1; then
    command -v cc
    return 0
  fi
  return 1
}

find_nvcc() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    echo /usr/local/cuda/bin/nvcc
    return 0
  fi
  return 1
}

if nvcc_path="$(find_nvcc)"; then
  CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "[starvla-qwen35-fastpath] using nvcc: ${nvcc_path}"
else
  echo "[starvla-qwen35-fastpath] nvcc not found; skipped optional Qwen3.5 CUDA fast-path packages." >&2
  echo "[starvla-qwen35-fastpath] StarVLA runtime remains usable through the torch fallback path." >&2
  exit 0
fi

if cxx_path="$(find_cxx)" && cc_path="$(find_cc)"; then
  compiler_bin="${ENVSET_ROOT}/system_tools/compiler/bin"
  mkdir -p "${compiler_bin}"
  cat > "${compiler_bin}/clang++" <<EOF
#!/usr/bin/env bash
exec "${cxx_path}" "\$@"
EOF
  cat > "${compiler_bin}/clang" <<EOF
#!/usr/bin/env bash
exec "${cc_path}" "\$@"
EOF
  chmod +x "${compiler_bin}/clang++" "${compiler_bin}/clang"
  export PATH="${compiler_bin}:${PATH}"
  export CC="${cc_path}"
  export CXX="${cxx_path}"
  export CUDAHOSTCXX="${cxx_path}"
  echo "[starvla-qwen35-fastpath] using C compiler: ${CC}"
  echo "[starvla-qwen35-fastpath] using C++ compiler: ${CXX}"
  echo "[starvla-qwen35-fastpath] installed clang/clang++ compatibility wrappers: ${compiler_bin}"
else
  echo "[starvla-qwen35-fastpath] no usable C/C++ compiler found before CUDA build." >&2
  echo "[starvla-qwen35-fastpath] Install or expose gcc/g++ or clang/clang++ in PATH, then rerun." >&2
  exit 1
fi

export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export STARVLA_CUDA_ARCHS="${STARVLA_CUDA_ARCHS:-90}"
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# causal-conv1d tries GitHub release wheels unless this is forced. GPU nodes are
# offline, so the only valid path here is a local CUDA source build.
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_CXX11_ABI="${CAUSAL_CONV1D_FORCE_CXX11_ABI:-FALSE}"

mkdir -p "${LOG_DIR}"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/starvla-qwen35-fastpath.XXXXXX")"
cleanup() {
  rm -rf "${build_root}"
}
trap cleanup EXIT

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[starvla-qwen35-fastpath] ${name} started; log=${log_file}"
  set +e
  "$@" > >(tee -a "${log_file}") 2>&1 &
  local pid=$!
  local start_ts
  start_ts="$(date +%s)"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
    if kill -0 "${pid}" 2>/dev/null; then
      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_ts))
      echo "[starvla-qwen35-fastpath] ${name} still running after ${elapsed}s; log=${log_file}"
      tail -n 5 "${log_file}" 2>/dev/null || true
    fi
  done
  wait "${pid}"
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "[starvla-qwen35-fastpath] ${name} failed with status ${status}; tail=${log_file}" >&2
    tail -n 80 "${log_file}" >&2 || true
    return "${status}"
  fi
  echo "[starvla-qwen35-fastpath] ${name} completed; log=${log_file}"
}

patched_causal_src="${build_root}/causal_conv1d_src"
mkdir -p "${patched_causal_src}"
tar --no-same-owner -xf "${causal_src}" -C "${patched_causal_src}" --strip-components=1
"${PYTHON}" - "${patched_causal_src}/setup.py" "${STARVLA_CUDA_ARCHS}" <<'PY'
from pathlib import Path
import sys

setup_py = Path(sys.argv[1])
archs = [item.strip().replace(".", "") for item in sys.argv[2].split(",") if item.strip()]
if not archs:
    archs = ["90"]
text = setup_py.read_text(encoding="utf-8")
start_marker = '        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_75,code=sm_75")'
end_marker = '        if bare_metal_version >= Version("13.0"):'
start = text.index(start_marker)
end = text.index(end_marker, start)
block = "".join(
    f'        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_{arch},code=sm_{arch}")\n'
    for arch in archs
)
setup_py.write_text(text[:start] + block + text[end:], encoding="utf-8")
print(f"Patched causal-conv1d CUDA arch list to: {','.join(archs)}")
PY

echo "[starvla-qwen35-fastpath] build settings: MAX_JOBS=${MAX_JOBS}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, STARVLA_CUDA_ARCHS=${STARVLA_CUDA_ARCHS}"
if causal_conv1d_ok; then
  echo "[starvla-qwen35-fastpath] causal_conv1d already import OK; skipped CUDA rebuild."
else
  run_logged causal_conv1d_build "${PYTHON}" -m pip install --no-index --no-deps --no-build-isolation --no-cache-dir "${patched_causal_src}"
fi
if fla_runtime_ok; then
  echo "[starvla-qwen35-fastpath] FLA core runtime already import OK; skipped reinstall."
else
  install_fla_stack || true
fi
if causal_conv1d_ok; then
  echo "causal_conv1d import OK"
else
  echo "[starvla-qwen35-fastpath] causal_conv1d failed to import after install." >&2
  exit 1
fi
if fla_runtime_ok; then
  echo "FLA core runtime import OK"
else
  echo "[starvla-qwen35-fastpath] FLA core runtime is still unavailable; Qwen3.5 will use the Transformers torch fallback." >&2
fi
ensure_transformers_runtime
