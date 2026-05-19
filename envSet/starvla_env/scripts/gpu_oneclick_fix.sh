#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="$(starvla_default_env_dir "${REPO_ROOT}")"
RUN_CHECK=1
NUM_GPUS="${STARVLA_NUM_GPUS:-8}"
INSTALL_QWEN35_FASTPATH="${STARVLA_INSTALL_QWEN35_FASTPATH:-0}"
INSTALL_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/gpu_oneclick_fix.sh [options]

Offline repair for the GPU zone. It refreshes the StarVLA Python env,
tmux wrapper, NCCL perf wrappers, and StarVLA H200 code overlay without
using proxy or network access, then runs check/perf-check.

Options:
  --repo-root PATH      StarVLA repo root. Default: auto-detect.
  --env-dir PATH        Python env dir. Default: $REPO_ROOT/env/starvla-py310.
  --num-gpus N          Number of GPUs expected by perf-check. Default: 8.
  --force-restore       Restore the packaged env snapshot before patching.
  --force-reinstall     Reinstall Python packages even when the env is already healthy.
  --install-qwen35-fastpath
                        Also compile/install Qwen3.5 fast-path CUDA packages
                        from offline source archives. Best run on the H200 node.
  --skip-check          Only repair files; do not run check/perf-check.
  -h, --help            Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$(cd "$2" && pwd)"
      ENV_DIR="$(starvla_default_env_dir "${REPO_ROOT}")"
      INSTALL_ARGS+=("--repo-root" "${REPO_ROOT}")
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      INSTALL_ARGS+=("--env-dir" "${ENV_DIR}")
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --force-restore)
      INSTALL_ARGS+=("--force-restore")
      shift
      ;;
    --force-reinstall)
      INSTALL_ARGS+=("--force-reinstall")
      shift
      ;;
    --install-qwen35-fastpath)
      INSTALL_QWEN35_FASTPATH=1
      shift
      ;;
    --skip-check)
      RUN_CHECK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"

ensure_transformers_runtime() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  if "${python}" - <<'PY' >/dev/null 2>&1
import transformers

for attr in ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    getattr(transformers, attr)
PY
  then
    echo "[starvla-gpu-fix] Transformers Qwen runtime OK."
    return 0
  fi

  echo "[starvla-gpu-fix] Transformers Qwen runtime is not healthy; refreshing overlay only."
  "${SCRIPT_DIR}/apply_transformers_5_2_overlay.sh" "${ENV_DIR}"
  "${python}" - <<'PY'
import transformers

for attr in ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    getattr(transformers, attr)
print("Transformers Qwen runtime OK")
PY
}

echo "[starvla-gpu-fix] repo=${REPO_ROOT}"
echo "[starvla-gpu-fix] env=${ENV_DIR}"
echo "[starvla-gpu-fix] envset=${ENVSET_ROOT}"

"${SCRIPT_DIR}/install_offline.sh" --skip-check "${INSTALL_ARGS[@]}"
if [ "${INSTALL_QWEN35_FASTPATH}" = "1" ]; then
  if ! "${SCRIPT_DIR}/install_qwen35_fastpath.sh" "${ENV_DIR}"; then
    echo "[starvla-gpu-fix] Qwen3.5 fast-path install failed; runtime still works via torch fallback." >&2
  fi
fi
ensure_transformers_runtime

export STARVLA_REPO_ROOT="${REPO_ROOT}"
export STARVLA_ENVSET_ROOT="${ENVSET_ROOT}"
export STARVLA_ENV_DIR="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${ENVSET_ROOT}/performance_tools/bin:${ENVSET_ROOT}/system_tools/tmux/bin:${PATH}"

if [ "${RUN_CHECK}" -eq 0 ]; then
  echo "[starvla-gpu-fix] repair complete; checks skipped."
  exit 0
fi

cd "${REPO_ROOT}"
bash interaction/bin/starvla-interact.sh check

if command -v nvidia-smi >/dev/null 2>&1; then
  bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus "${NUM_GPUS}"
else
  echo "[starvla-gpu-fix] nvidia-smi not visible; skipped GPU perf-check."
fi

echo "[starvla-gpu-fix] repair and checks completed."
