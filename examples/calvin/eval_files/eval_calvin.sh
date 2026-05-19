#!/bin/bash
set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${REPO_ROOT}"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
DATA_ROOT=${STARVLA_DATA_ROOT:-$(dirname "${REPO_ROOT}")/data/starvla}
DEFAULT_CKPT=${DEFAULT_CKPT:-${DATA_ROOT}/checkpoints/qwen35_0_8b-QwenPI-calvin_task_ABC/interactive_calvin_0519_071115/final_model/pytorch_model.pt}
CALVIN_HOME=${CALVIN_HOME:-${REPO_ROOT}/playground/Code/calvin}
export PYTHONPATH=${REPO_ROOT}:${CALVIN_HOME}:${CALVIN_HOME}/calvin_env:${CALVIN_HOME}/calvin_env/tacto:${CALVIN_HOME}/calvin_models:${PYTHONPATH:-}
export calvin_python=${calvin_python:-${REPO_ROOT}/env/starvla-py310/bin/python}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/starvla-mplconfig-${USER:-user}}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/starvla-cache-${USER:-user}}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-${XDG_CACHE_HOME}/mesa_shader_cache}
unset DEBUG
export STARVLA_ENABLE_DEBUGPY=${STARVLA_ENABLE_DEBUGPY:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export STARVLA_CALVIN_RENDER_BACKEND=${STARVLA_CALVIN_RENDER_BACKEND:-auto}
export STARVLA_CALVIN_STATE_DIM=${STARVLA_CALVIN_STATE_DIM:-auto}
export STARVLA_CALVIN_STATE_SLICE=${STARVLA_CALVIN_STATE_SLICE:-first}

_starvla_calvin_use_egl() {
    export STARVLA_CALVIN_RENDER_BACKEND=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_GL=egl
    export LIBGL_ALWAYS_SOFTWARE=0
}

_starvla_calvin_use_direct() {
    export STARVLA_CALVIN_RENDER_BACKEND=direct
    export PYOPENGL_PLATFORM=osmesa
    export MUJOCO_GL=osmesa
    export LIBGL_ALWAYS_SOFTWARE=1
    unset EGL_VISIBLE_DEVICE
    unset EGL_VISIBLE_DEVICES
}

_starvla_calvin_egl_preflight() {
    PYOPENGL_PLATFORM=egl MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=0 "${calvin_python}" - <<'PY' >/dev/null 2>&1
import os
import pkgutil
import pybullet as p

render_gpu = os.environ.get("STARVLA_CALVIN_RENDER_GPU", "").strip()
if render_gpu:
    try:
        from calvin_env.utils.utils import get_egl_device_id

        egl_id = get_egl_device_id(int(render_gpu.split(",")[0]))
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        os.environ["EGL_VISIBLE_DEVICE"] = str(egl_id)
    except Exception:
        pass

cid = p.connect(p.DIRECT, options="--width=16 --height=16")
try:
    egl = pkgutil.get_loader("eglRenderer")
    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin") if egl else p.loadPlugin("eglRendererPlugin")
    if plugin < 0:
        raise SystemExit(1)
finally:
    p.disconnect(cid)
PY
}

_starvla_calvin_cached_egl_preflight() {
    local gpu_key cache_dir status_file lock_dir waited result
    gpu_key=${STARVLA_CALVIN_RENDER_GPU:-auto}
    gpu_key=${gpu_key//[^A-Za-z0-9_.-]/_}
    if [[ -n "${STARVLA_JOB_DIR:-}" ]]; then
        cache_dir="${STARVLA_JOB_DIR}/calvin_eval/egl_preflight"
    else
        cache_dir="${XDG_CACHE_HOME}/starvla_calvin_egl_preflight"
    fi
    status_file="${cache_dir}/gpu_${gpu_key}.status"
    lock_dir="${cache_dir}/gpu_${gpu_key}.lock"
    mkdir -p "${cache_dir}"

    if [[ -s "${status_file}" ]]; then
        cat "${status_file}"
        return 0
    fi

    if mkdir "${lock_dir}" 2>/dev/null; then
        result=direct
        if _starvla_calvin_egl_preflight; then
            result=egl
        fi
        printf '%s\n' "${result}" > "${status_file}.$$"
        mv "${status_file}.$$" "${status_file}"
        rmdir "${lock_dir}" 2>/dev/null || true
        cat "${status_file}"
        return 0
    fi

    waited=0
    while [[ "${waited}" -lt 120 ]]; do
        if [[ -s "${status_file}" ]]; then
            cat "${status_file}"
            return 0
        fi
        sleep 0.5
        waited=$((waited + 1))
    done

    if _starvla_calvin_egl_preflight; then
        echo egl
    else
        echo direct
    fi
}

case "${STARVLA_CALVIN_RENDER_BACKEND}" in
    egl|gpu|cuda|fast)
        _starvla_calvin_use_egl
        ;;
    direct|osmesa|safe|cpu)
        _starvla_calvin_use_direct
        ;;
    auto|"")
        if [[ "$(_starvla_calvin_cached_egl_preflight)" == "egl" ]]; then
            _starvla_calvin_use_egl
        else
            echo "[starvla-calvin-eval] EGL preflight failed or was cached as unavailable; falling back to direct/osmesa rendering."
            _starvla_calvin_use_direct
        fi
        ;;
    *)
        echo "[starvla-calvin-eval] Unknown STARVLA_CALVIN_RENDER_BACKEND=${STARVLA_CALVIN_RENDER_BACKEND}; using direct/osmesa."
        _starvla_calvin_use_direct
        ;;
esac

host=${host:-127.0.0.1}
base_port=${base_port:-5694}
unnorm_key=${unnorm_key:-franka}
your_ckpt=${your_ckpt:-${CKPT_PATH:-${DEFAULT_CKPT}}}
dataset_path=${dataset_path:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d/validation}
calvin_config_path=${calvin_config_path:-${CALVIN_HOME}/calvin_models/conf}
eval_sequences_path=${eval_sequences_path:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}
num_sequences=${num_sequences:-1000}
max_steps_per_task=${max_steps_per_task:-360}
num_ddim_steps=${num_ddim_steps:-10}
sequence_start=${sequence_start:-0}
sequence_end=${sequence_end:--1}
sequence_stride=${sequence_stride:-1}
eval_log_dir=${eval_log_dir:-${STARVLA_JOB_DIR:-$(pwd)/logs/calvin_eval_$(date +"%Y%m%d_%H%M%S")}}

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

if [[ ! -f "${your_ckpt}" ]]; then
    echo "[starvla-calvin-eval] Checkpoint not found: ${your_ckpt}" >&2
    echo "[starvla-calvin-eval] Set CKPT_PATH=/path/to/pytorch_model.pt or your_ckpt=/path/to/pytorch_model.pt" >&2
    exit 1
fi

mkdir -p "${eval_log_dir}" "${MPLCONFIGDIR}" "${MESA_SHADER_CACHE_DIR}"

echo "[starvla-calvin-eval] ckpt=${your_ckpt}"
echo "[starvla-calvin-eval] dataset_path=${dataset_path}"
echo "[starvla-calvin-eval] calvin_config_path=${calvin_config_path}"
echo "[starvla-calvin-eval] eval_sequences_path=${eval_sequences_path}"
echo "[starvla-calvin-eval] num_sequences=${num_sequences}"
echo "[starvla-calvin-eval] max_steps_per_task=${max_steps_per_task}"
echo "[starvla-calvin-eval] num_ddim_steps=${num_ddim_steps}"
echo "[starvla-calvin-eval] sequence_start=${sequence_start}"
echo "[starvla-calvin-eval] sequence_end=${sequence_end}"
echo "[starvla-calvin-eval] sequence_stride=${sequence_stride}"
echo "[starvla-calvin-eval] eval_log_dir=${eval_log_dir}"
echo "[starvla-calvin-eval] python=${calvin_python}"
echo "[starvla-calvin-eval] CALVIN_HOME=${CALVIN_HOME}"
echo "[starvla-calvin-eval] render_backend=${STARVLA_CALVIN_RENDER_BACKEND}"
echo "[starvla-calvin-eval] render_gpu=${STARVLA_CALVIN_RENDER_GPU:-auto}"
echo "[starvla-calvin-eval] PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-}"
echo "[starvla-calvin-eval] MUJOCO_GL=${MUJOCO_GL:-}"
echo "[starvla-calvin-eval] state_dim=${STARVLA_CALVIN_STATE_DIM}"
echo "[starvla-calvin-eval] state_slice=${STARVLA_CALVIN_STATE_SLICE}"

"${calvin_python}" ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.host "$host" \
    --args.port "$base_port" \
    --args.dataset_path "${dataset_path}" \
    --args.calvin_config_path "${calvin_config_path}" \
    --args.eval_sequences_path "${eval_sequences_path}" \
    --args.eval_log_dir "${eval_log_dir}" \
    --args.num_sequences "${num_sequences}" \
    --args.num_ddim_steps "${num_ddim_steps}" \
    --args.max_steps_per_task "${max_steps_per_task}" \
    --args.sequence_start "${sequence_start}" \
    --args.sequence_end "${sequence_end}" \
    --args.sequence_stride "${sequence_stride}" \
    "$@"
