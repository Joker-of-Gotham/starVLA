#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="${1:-$(starvla_default_env_dir "${REPO_ROOT}")}"
PYTHON="${ENV_DIR}/bin/python"

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"
export STARVLA_REPO_ROOT="${REPO_ROOT}"
export STARVLA_ENVSET_ROOT="${ENVSET_ROOT}"
export STARVLA_ENV_DIR="${ENV_DIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/playground/Code/calvin:${REPO_ROOT}/playground/Code/calvin/calvin_env:${REPO_ROOT}/playground/Code/calvin/calvin_env/tacto:${REPO_ROOT}/playground/Code/calvin/calvin_models${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/starvla-mplconfig-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/starvla-cache-${USER:-user}}"
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-${XDG_CACHE_HOME}/mesa_shader_cache}"
mkdir -p "${MPLCONFIGDIR}" "${MESA_SHADER_CACHE_DIR}"

if [ ! -x "${PYTHON}" ]; then
  echo "FAIL: python not found: ${PYTHON}" >&2
  exit 1
fi

echo "repo=${REPO_ROOT}"
echo "env=${ENV_DIR}"
echo "python=$("${PYTHON}" -c 'import sys; print(sys.executable)')"
echo "tmux=$(command -v tmux || true)"
tmux -V
echo "all_reduce_perf=$(command -v all_reduce_perf || true)"
if command -v all_reduce_perf >/dev/null 2>&1; then
  all_reduce_perf -h >/dev/null
fi

"${PYTHON}" - <<'PY'
import importlib
import json
import os
from pathlib import Path
import sys

mods = [
    "torch",
    "flash_attn",
    "accelerate",
    "transformers",
    "diffusers",
    "decord",
    "pytorch3d",
    "rich",
    "omegaconf",
    "hydra",
    "pybullet",
    "gym",
    "quaternion",
    "moviepy.editor",
    "numba",
    "calvin_env",
]
required_runtime_capabilities = {
    "transformers.Qwen3_5ForConditionalGeneration": ("transformers", "Qwen3_5ForConditionalGeneration"),
    "transformers.Qwen3VLForConditionalGeneration": ("transformers", "Qwen3VLForConditionalGeneration"),
}

out = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "proxy_clean": not any(os.environ.get(k) for k in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")),
}

failed = []
for name in mods:
    try:
        module = importlib.import_module(name)
        out[name] = getattr(module, "__version__", "ok")
    except Exception as exc:
        out[name] = f"FAIL: {exc!r}"
        failed.append(name)

for capability, (module_name, attr_name) in required_runtime_capabilities.items():
    try:
        module = importlib.import_module(module_name)
        try:
            getattr(module, attr_name)
            out[capability] = "ok"
        except AttributeError:
            out[capability] = "FAIL: missing"
            failed.append(capability)
        except Exception as exc:
            cause = getattr(exc, "__cause__", None)
            detail = repr(exc)
            if cause is not None:
                detail += f"; cause={cause!r}"
            out[capability] = f"FAIL: {detail}"
            failed.append(capability)
    except Exception as exc:
        out[capability] = f"FAIL: {exc!r}"
        failed.append(capability)

try:
    import torch

    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    out["gpu_count"] = torch.cuda.device_count()
except Exception as exc:
    out["torch_runtime"] = f"FAIL: {exc!r}"
    failed.append("torch_runtime")

repo = Path(os.environ["STARVLA_REPO_ROOT"])
for rel in [
    "interaction/core/perf.py",
    "interaction/config/accelerate_h200_zero2.yaml",
    "interaction/config/ds_h200_zero2.json",
]:
    exists = (repo / rel).exists()
    out[rel] = exists
    if not exists:
        failed.append(rel)

try:
    from interaction.core.perf import resolve_perf_settings

    perf = resolve_perf_settings(profile="h200_8gpu", selected_gpus=list(range(8)))
    out["perf_profile_import"] = perf.resolved_profile
    out["perf_accelerate_config"] = str(perf.accelerate_config)
except Exception as exc:
    out["perf_profile_import"] = f"FAIL: {exc!r}"
    failed.append("interaction.core.perf")

print(json.dumps(out, ensure_ascii=False, indent=2))
if failed:
    raise SystemExit("Import check failed: " + ", ".join(failed))
PY

if [ -f "${REPO_ROOT}/bar/check_action_heads.py" ]; then
  (cd "${REPO_ROOT}" && "${PYTHON}" bar/check_action_heads.py)
fi

if [ -x "${REPO_ROOT}/interaction/bin/starvla-interact.sh" ]; then
  (cd "${REPO_ROOT}" && bash interaction/bin/starvla-interact.sh check >/tmp/starvla_interaction_offline_check.txt 2>&1)
  tail -n 80 /tmp/starvla_interaction_offline_check.txt
  (cd "${REPO_ROOT}" && bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8 >/tmp/starvla_interaction_perf_check.txt 2>&1)
  tail -n 80 /tmp/starvla_interaction_perf_check.txt
fi

echo "Offline environment check passed."
