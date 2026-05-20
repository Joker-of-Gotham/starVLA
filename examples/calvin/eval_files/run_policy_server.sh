#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${REPO_ROOT}"
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
DATA_ROOT=${STARVLA_DATA_ROOT:-$(dirname "${REPO_ROOT}")/data/starvla}
DEFAULT_CKPT=${DEFAULT_CKPT:-${DATA_ROOT}/checkpoints/qwen35_0_8b-QwenPI-calvin_task_ABC/interactive_calvin_0519_071115/final_model/pytorch_model.pt}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export star_vla_python=${star_vla_python:-${STAR_VLA_PYTHON:-${REPO_ROOT}/env/starvla-py310/bin/python}}
your_ckpt=${your_ckpt:-${CKPT_PATH:-${DEFAULT_CKPT}}}
gpu_id=${gpu_id:-${GPU_ID:-0}}
port=${port:-${PORT:-5694}}
idle_timeout=${idle_timeout:-${IDLE_TIMEOUT:--1}}
use_bf16=${use_bf16:-${USE_BF16:-1}}
policy_device=${policy_device:-${STARVLA_POLICY_DEVICE:-cuda}}
retry_on_137=${retry_on_137:-${STARVLA_RETRY_ON_137:-0}}
server_concurrency=${server_concurrency:-${STARVLA_SERVER_INFERENCE_CONCURRENCY:-1}}
unset DEBUG
export STARVLA_ENABLE_DEBUGPY=${STARVLA_ENABLE_DEBUGPY:-0}
export STARVLA_POLICY_STATE_ADAPT=${STARVLA_POLICY_STATE_ADAPT:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export STARVLA_CLIENT_CONNECT_TIMEOUT=${STARVLA_CLIENT_CONNECT_TIMEOUT:-900}
export STARVLA_CLIENT_FAIL_ON_CONNECT_TIMEOUT=${STARVLA_CLIENT_FAIL_ON_CONNECT_TIMEOUT:-1}
export STARVLA_CLIENT_REQUEST_TIMEOUT=${STARVLA_CLIENT_REQUEST_TIMEOUT:-300}
export STARVLA_CLIENT_FAIL_ON_REQUEST_TIMEOUT=${STARVLA_CLIENT_FAIL_ON_REQUEST_TIMEOUT:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
################# star Policy Server ######################

if [[ ! -f "${your_ckpt}" ]]; then
    echo "Checkpoint not found: ${your_ckpt}" >&2
    exit 1
fi

echo "[starvla-policy-server] ckpt=${your_ckpt}"
echo "[starvla-policy-server] gpu_id=${gpu_id}"
echo "[starvla-policy-server] port=${port}"
echo "[starvla-policy-server] python=${star_vla_python}"
echo "[starvla-policy-server] state_adapt=${STARVLA_POLICY_STATE_ADAPT}"
echo "[starvla-policy-server] device=${policy_device}"
echo "[starvla-policy-server] use_bf16=${use_bf16}"
echo "[starvla-policy-server] max_concurrency=${server_concurrency}"

server_cmd=(
    "${star_vla_python}"
    deployment/model_server/server_policy.py
    --ckpt_path "${your_ckpt}"
    --port "${port}"
    --idle_timeout "${idle_timeout}"
    --device "${policy_device}"
    --max_concurrency "${server_concurrency}"
)
if [[ "${use_bf16}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    server_cmd+=(--use_bf16)
fi

run_server() {
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${server_cmd[@]}"
}

set +e
run_server
status=$?
set -e

if [[ "${status}" -eq 137 && "${retry_on_137}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && "${policy_device}" != "cpu" ]]; then
    echo "[starvla-policy-server] server was SIGKILLed with status 137, usually by OOM/resource limits." >&2
    echo "[starvla-policy-server] Retrying on CPU to keep the evaluation pipeline alive. Set STARVLA_RETRY_ON_137=0 to disable." >&2
    policy_device=cpu
    export STARVLA_POLICY_DEVICE=cpu
    server_cmd=(
        "${star_vla_python}"
        deployment/model_server/server_policy.py
        --ckpt_path "${your_ckpt}"
        --port "${port}"
        --idle_timeout "${idle_timeout}"
        --device "${policy_device}"
        --max_concurrency "${server_concurrency}"
    )
    if [[ "${use_bf16}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        server_cmd+=(--use_bf16)
    fi
    set +e
    run_server
    status=$?
    set -e
fi

if [[ "${status}" -eq 137 ]]; then
    echo "[starvla-policy-server] server is still being killed with status 137 after retry." >&2
    echo "[starvla-policy-server] This is a resource-limit/OOM kill, not a scheduler_config warning. Use a smaller checkpoint or allocate more memory/GPU." >&2
fi

exit "${status}"

# #################################
