#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="$(starvla_default_env_dir "${REPO_ROOT}")"
ARCHIVE="${ENVSET_ROOT}/archives/starvla-py310-env.tar"
FLASH_WHEEL="$(find "${ENVSET_ROOT}/wheelhouse/flash_attn" -maxdepth 1 -name 'flash_attn-2.7.4.post1*.whl' | head -n 1)"
CALVIN_EVAL_WHEELHOUSE="${ENVSET_ROOT}/wheelhouse/calvin_eval"
FORCE_RESTORE=0
FORCE_REINSTALL=0
SKIP_CHECK=0
SKIP_INTERACTION_OVERLAY=0

usage() {
  cat <<'EOF'
Usage: scripts/install_offline.sh [options]

Options:
  --repo-root PATH      StarVLA repo root. Default: auto-detect, including /26220447/starVLA.
  --env-dir PATH        Python env dir. Default: $REPO_ROOT/env/starvla-py310.
  --force-restore       Always restore the packaged env archive.
  --force-reinstall     Reinstall flash-attn, Transformers overlay, and editable StarVLA even if healthy.
  --skip-interaction-overlay
                        Do not refresh interaction/ from the packaged H200 overlay.
  --skip-check          Skip final import/action-head checks.
  -h, --help            Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$(cd "$2" && pwd)"
      ENV_DIR="$(starvla_default_env_dir "${REPO_ROOT}")"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --force-restore)
      FORCE_RESTORE=1
      shift
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    --skip-interaction-overlay)
      SKIP_INTERACTION_OVERLAY=1
      shift
      ;;
    --skip-check)
      SKIP_CHECK=1
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
export STARVLA_REPO_ROOT="${REPO_ROOT}"
export STARVLA_ENVSET_ROOT="${ENVSET_ROOT}"
export STARVLA_ENV_DIR="${ENV_DIR}"

install_tmux_link() {
  mkdir -p "${ENV_DIR}/bin"
  ln -sfn "${ENVSET_ROOT}/system_tools/tmux/bin/tmux" "${ENV_DIR}/bin/tmux"
  "${ENV_DIR}/bin/tmux" -V >/dev/null
}

install_perf_tools() {
  "${SCRIPT_DIR}/install_perf_tools.sh" "${ENV_DIR}"
}

install_interaction_overlay() {
  local overlay="${ENVSET_ROOT}/repo_overlay"
  [ "${SKIP_INTERACTION_OVERLAY}" -eq 0 ] || return 0
  [ -d "${overlay}" ] || return 0
  cp -a "${overlay}/." "${REPO_ROOT}/"
  chmod +x "${REPO_ROOT}/interaction/bin/starvla-interact.sh" 2>/dev/null || true
  chmod +x "${REPO_ROOT}/interaction/bin/starvla-gpu-fix.sh" 2>/dev/null || true
}

install_transformers_overlay() {
  "${SCRIPT_DIR}/apply_transformers_5_2_overlay.sh" "${ENV_DIR}"
}

calvin_pythonpath() {
  printf '%s:%s:%s:%s:%s' \
    "${REPO_ROOT}" \
    "${REPO_ROOT}/playground/Code/calvin" \
    "${REPO_ROOT}/playground/Code/calvin/calvin_env" \
    "${REPO_ROOT}/playground/Code/calvin/calvin_env/tacto" \
    "${REPO_ROOT}/playground/Code/calvin/calvin_models"
}

quick_env_check() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  PYTHONPATH="$(calvin_pythonpath)${PYTHONPATH:+:${PYTHONPATH}}" "${python}" - <<'PY' >/dev/null 2>&1
import importlib
import importlib.metadata as metadata
import os
from pathlib import Path

repo = Path(os.environ["STARVLA_REPO_ROOT"]).resolve()

flash_attn = importlib.import_module("flash_attn")
if not str(getattr(flash_attn, "__version__", "")).startswith("2.7.4"):
    raise SystemExit("flash_attn version mismatch")

transformers = importlib.import_module("transformers")
if metadata.version("transformers") != "5.2.0":
    raise SystemExit("transformers version mismatch")
for attr in ("Qwen3_5ForConditionalGeneration", "Qwen3VLForConditionalGeneration"):
    if not hasattr(transformers, attr):
        raise SystemExit(f"missing transformers.{attr}")

starvla = importlib.import_module("starVLA")
starvla_file = Path(getattr(starvla, "__file__", "")).resolve()
if repo not in starvla_file.parents:
    raise SystemExit(f"starVLA is not editable from repo: {starvla_file}")

from interaction.core.perf import resolve_perf_settings
from interaction.core.throughput import resolve_throughput_settings
for module_name in ("hydra", "pybullet", "gym", "quaternion", "moviepy.editor", "numba", "calvin_env"):
    importlib.import_module(module_name)

resolve_perf_settings(profile="balanced", selected_gpus=[])
resolve_throughput_settings(
    profile="none",
    selected_gpus=[],
    model_key="qwen35_0_8b",
    framework="QwenPI",
    mode="vla",
    preset_vla_batch=1,
    preset_vlm_batch=1,
    explicit_vla_batch=None,
    explicit_vlm_batch=None,
    explicit_workers=None,
    explicit_prefetch=None,
    explicit_pin_memory=None,
    explicit_persistent_workers=None,
)
PY
}

calvin_eval_deps_ok() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  PYTHONPATH="$(calvin_pythonpath)${PYTHONPATH:+:${PYTHONPATH}}" "${python}" - <<'PY' >/dev/null 2>&1
import importlib

for module_name in ("hydra", "pybullet", "gym", "quaternion", "moviepy.editor", "numba", "calvin_env"):
    importlib.import_module(module_name)
PY
}

install_calvin_eval_deps() {
  local python="${ENV_DIR}/bin/python"
  if [ "${FORCE_REINSTALL}" -eq 0 ] && calvin_eval_deps_ok; then
    echo "[starvla-offline] CALVIN eval deps already OK; skipped reinstall."
    return 0
  fi
  if [ ! -d "${CALVIN_EVAL_WHEELHOUSE}" ]; then
    echo "Missing CALVIN eval offline wheelhouse: ${CALVIN_EVAL_WHEELHOUSE}" >&2
    return 1
  fi
  "${python}" -m pip install --no-index --only-binary=:all: --find-links "${CALVIN_EVAL_WHEELHOUSE}" \
    "numpy<2" hydra-core hydra-colorlog pybullet moviepy gym numpy-quaternion cloudpickle
}

flash_attn_ok() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  "${python}" - <<'PY' >/dev/null 2>&1
import flash_attn
if not str(getattr(flash_attn, "__version__", "")).startswith("2.7.4"):
    raise SystemExit(1)
PY
}

transformers_overlay_ok() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  "${python}" - <<'PY' >/dev/null 2>&1
import importlib.metadata as metadata
import transformers
assert metadata.version("transformers") == "5.2.0"
assert hasattr(transformers, "Qwen3_5ForConditionalGeneration")
assert hasattr(transformers, "Qwen3VLForConditionalGeneration")
PY
}

editable_starvla_ok() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${python}" - <<'PY' >/dev/null 2>&1
import os
from pathlib import Path
import starVLA

repo = Path(os.environ["STARVLA_REPO_ROOT"]).resolve()
starvla_file = Path(getattr(starVLA, "__file__", "")).resolve()
if repo not in starvla_file.parents:
    raise SystemExit(1)
PY
}

patch_existing_env() {
  local python="${ENV_DIR}/bin/python"
  [ -x "${python}" ] || return 1
  if [ "${FORCE_REINSTALL}" -eq 0 ] && flash_attn_ok; then
    echo "[starvla-offline] flash-attn already OK; skipped reinstall."
  else
    "${python}" -m pip install --no-index --no-deps --force-reinstall "${FLASH_WHEEL}"
  fi
  if [ "${FORCE_REINSTALL}" -eq 0 ] && transformers_overlay_ok; then
    echo "[starvla-offline] Transformers 5.2 overlay already OK; skipped overlay."
  else
    install_transformers_overlay
  fi
  if [ -f "${REPO_ROOT}/pyproject.toml" ] || [ -f "${REPO_ROOT}/setup.py" ]; then
    if [ "${FORCE_REINSTALL}" -eq 0 ] && editable_starvla_ok; then
      echo "[starvla-offline] editable StarVLA already points to repo; skipped editable install."
    else
      "${python}" -m pip install --no-index --no-deps --no-build-isolation -e "${REPO_ROOT}" || true
    fi
  fi
  install_calvin_eval_deps
  install_tmux_link
  install_perf_tools
  install_interaction_overlay
}

restore_archive() {
  local parent backup tmp
  parent="$(dirname "${ENV_DIR}")"
  mkdir -p "${parent}"
  if [ -e "${ENV_DIR}" ]; then
    backup="${ENV_DIR}.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Backing up existing env to ${backup}"
    mv "${ENV_DIR}" "${backup}"
  fi
  tmp="${parent}/.starvla_restore.$$"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"
  tar --no-same-owner -xf "${ARCHIVE}" -C "${tmp}"
  mv "${tmp}/starvla-py310" "${ENV_DIR}"
  rmdir "${tmp}"
  patch_existing_env
}

if [ "${FORCE_RESTORE}" -eq 0 ] && [ "${FORCE_REINSTALL}" -eq 0 ] && quick_env_check; then
  echo "[starvla-offline] existing env is healthy; skipped package reinstall."
  install_tmux_link
  install_perf_tools
  install_interaction_overlay
  if [ "${SKIP_CHECK}" -eq 0 ]; then
    "${SCRIPT_DIR}/check_offline_env.sh" "${ENV_DIR}"
  fi
  echo "Offline env is ready: ${ENV_DIR}"
  exit 0
fi

if [ ! -f "${ARCHIVE}" ]; then
  echo "Missing env archive: ${ARCHIVE}" >&2
  exit 1
fi
if [ -z "${FLASH_WHEEL}" ] || [ ! -f "${FLASH_WHEEL}" ]; then
  echo "Missing flash-attn offline wheel under ${ENVSET_ROOT}/wheelhouse/flash_attn" >&2
  exit 1
fi

if [ -f "${ARCHIVE}.sha256" ]; then
  (cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${ARCHIVE}.sha256")")
fi

if [ "${FORCE_RESTORE}" -eq 0 ] && [ -x "${ENV_DIR}/bin/python" ]; then
  echo "Patching existing env only where needed: ${ENV_DIR}"
  if patch_existing_env; then
    if [ "${SKIP_CHECK}" -eq 1 ] || "${SCRIPT_DIR}/check_offline_env.sh" "${ENV_DIR}"; then
      echo "Offline env is ready: ${ENV_DIR}"
      exit 0
    fi
  fi
  echo "Existing env did not pass check; restoring packaged snapshot."
fi

restore_archive

if [ "${SKIP_CHECK}" -eq 0 ]; then
  "${SCRIPT_DIR}/check_offline_env.sh" "${ENV_DIR}"
fi

echo "Offline env is ready: ${ENV_DIR}"
