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
unset DEBUG
export STARVLA_ENABLE_DEBUGPY=${STARVLA_ENABLE_DEBUGPY:-0}
export STARVLA_POLICY_STATE_ADAPT=${STARVLA_POLICY_STATE_ADAPT:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
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

server_cmd=(
    "${star_vla_python}"
    deployment/model_server/server_policy.py
    --ckpt_path "${your_ckpt}"
    --port "${port}"
    --idle_timeout "${idle_timeout}"
)
if [[ "${use_bf16}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    server_cmd+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${server_cmd[@]}"

# #################################
