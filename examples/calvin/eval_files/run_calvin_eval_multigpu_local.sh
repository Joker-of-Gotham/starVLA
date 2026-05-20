#!/bin/bash
set -euo pipefail

# Multi-GPU CALVIN evaluation:
# - starts one policy server per worker
# - splits official sequences into contiguous worker shards
# - waits on websocket handshakes instead of fixed sleeps
# - writes per-worker logs and a merged_results.json

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${REPO_ROOT}"

unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy

DATA_ROOT=${STARVLA_DATA_ROOT:-$(dirname "${REPO_ROOT}")/data/starvla}
DEFAULT_CKPT=${DEFAULT_CKPT:-${DATA_ROOT}/checkpoints/qwen35_0_8b-QwenPI-calvin_task_ABC/interactive_calvin_0519_071115/final_model/pytorch_model.pt}
CKPT_PATH=${CKPT_PATH:-${your_ckpt:-${DEFAULT_CKPT}}}
CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_HOME=${CALVIN_HOME:-${REPO_ROOT}/playground/Code/calvin}
CALVIN_ROOT=${CALVIN_ROOT:-${CALVIN_HOME}}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-${star_vla_python:-${REPO_ROOT}/env/starvla-py310/bin/python}}
CALVIN_PYTHON=${CALVIN_PYTHON:-${calvin_python:-${STAR_VLA_PYTHON}}}
CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}

NUM_GPUS=${NUM_GPUS:-1}
PROCS_PER_GPU=${PROCS_PER_GPU:-1}
NUM_WORKERS=${NUM_WORKERS:-$((NUM_GPUS * PROCS_PER_GPU))}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
BASE_PORT=${BASE_PORT:-5694}
START_GPU=${START_GPU:-0}
UNNORM_KEY=${UNNORM_KEY:-auto}
MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK:-${CALVIN_EP_LEN:-360}}
CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS:-10}
STAGGER_SERVER_SECONDS=${STAGGER_SERVER_SECONDS:-2}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
STARVLA_CALVIN_RENDER_BACKEND=${STARVLA_CALVIN_RENDER_BACKEND:-auto}
USE_BF16=${USE_BF16:-1}
POLICY_DEVICE=${POLICY_DEVICE:-${STARVLA_POLICY_DEVICE:-cuda}}
STARVLA_RETRY_ON_137=${STARVLA_RETRY_ON_137:-0}
SERVER_CONCURRENCY=${SERVER_CONCURRENCY:-${STARVLA_SERVER_INFERENCE_CONCURRENCY:-1}}

RUN_TAG=${RUN_TAG:-$(basename "$(dirname "$(dirname "${CKPT_PATH}")")")_$(basename "${CKPT_PATH}" .pt)}
EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/${RUN_TAG}_multigpu_$(date -u +%Y%m%d_%H%M%S)}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-${CALVIN_ROOT}/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}
EVAL_PY=${EVAL_PY:-examples/calvin/eval_files/eval_calvin.py}

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CALVIN_DATASET_PATH}/validation" ]]; then
  echo "CALVIN_DATASET_PATH must contain validation/: ${CALVIN_DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks" ]]; then
  echo "CALVIN_CONFIG_PATH must point to calvin_models/conf: ${CALVIN_CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_SEQUENCES_PATH}" ]]; then
  echo "EVAL_SEQUENCES_PATH not found: ${EVAL_SEQUENCES_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "Eval python file not found: ${EVAL_PY}" >&2
  exit 1
fi

"${STAR_VLA_PYTHON}" - "${BASE_PORT}" "${NUM_WORKERS}" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
num_workers = int(sys.argv[2])
busy = []
for port in range(base_port, base_port + num_workers):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()

if busy:
    print("Some policy-server ports are already in use:", file=sys.stderr)
    for port, err in busy:
        print(f"  port {port}: {err}", file=sys.stderr)
    print("Set BASE_PORT to a free range or stop the old eval/server process.", file=sys.stderr)
    sys.exit(1)
PY

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" - <<'PY' >/dev/null 2>&1
import msgpack
try:
    import websockets.sync.client
except Exception:
    import websocket
try:
    import cv2
except Exception:
    pass
PY
then
  echo "CALVIN_PYTHON is missing msgpack plus websockets.sync.client or websocket-client." >&2
  echo "CALVIN_PYTHON=${CALVIN_PYTHON}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/starvla-mplconfig-${USER:-user}}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/starvla-cache-${USER:-user}}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-${XDG_CACHE_HOME}/mesa_shader_cache}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export STARVLA_CLIENT_CONNECT_TIMEOUT=${STARVLA_CLIENT_CONNECT_TIMEOUT:-${SERVER_READY_TIMEOUT}}
export STARVLA_CLIENT_FAIL_ON_CONNECT_TIMEOUT=${STARVLA_CLIENT_FAIL_ON_CONNECT_TIMEOUT:-1}
export STARVLA_CLIENT_REQUEST_TIMEOUT=${STARVLA_CLIENT_REQUEST_TIMEOUT:-300}
export STARVLA_CLIENT_FAIL_ON_REQUEST_TIMEOUT=${STARVLA_CLIENT_FAIL_ON_REQUEST_TIMEOUT:-1}
export STARVLA_RETRY_ON_137
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p "${MPLCONFIGDIR}" "${MESA_SHADER_CACHE_DIR}"

{
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}"
  echo "CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}"
  echo "EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH}"
  echo "EVAL_PY=${EVAL_PY}"
  echo "EVAL_ROOT=${EVAL_ROOT}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "PROCS_PER_GPU=${PROCS_PER_GPU}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
  echo "BASE_PORT=${BASE_PORT}"
  echo "START_GPU=${START_GPU}"
  echo "MAX_STEPS_PER_TASK=${MAX_STEPS_PER_TASK}"
  echo "CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS}"
  echo "STARVLA_CALVIN_RENDER_BACKEND=${STARVLA_CALVIN_RENDER_BACKEND}"
  echo "POLICY_DEVICE=${POLICY_DEVICE}"
  echo "SERVER_CONCURRENCY=${SERVER_CONCURRENCY}"
  echo "SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT}"
  echo "STARVLA_CLIENT_CONNECT_TIMEOUT=${STARVLA_CLIENT_CONNECT_TIMEOUT}"
  echo "STARVLA_CLIENT_REQUEST_TIMEOUT=${STARVLA_CLIENT_REQUEST_TIMEOUT}"
  echo "STARVLA_RETRY_ON_137=${STARVLA_RETRY_ON_137}"
} | tee "${EVAL_ROOT}/run.info"

pids=()
eval_pids=()
cleanup() {
  for pid in "${eval_pids[@]:-}" "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

wait_for_policy_server() {
  local port="$1"
  PYTHONPATH="${REPO_ROOT}:${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" - "${port}" "${SERVER_READY_TIMEOUT}" <<'PY'
import sys

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

port = int(sys.argv[1])
timeout = float(sys.argv[2])
timeout = timeout if timeout > 0 else None
client = WebsocketClientPolicy("127.0.0.1", port, connect_timeout=timeout)
metadata = client.get_server_metadata()
client.close()
print(f"server ready on port {port}: {metadata}")
PY
}

start_eval_worker() {
  local worker="$1"
  local port="$2"
  local start="$3"
  local end="$4"
  local gpu="$5"
  local worker_dir="${EVAL_ROOT}/worker_${worker}"
  mkdir -p "${worker_dir}"
  (
    export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}:${CALVIN_ROOT}/calvin_env:${CALVIN_ROOT}/calvin_env/tacto:${CALVIN_ROOT}/calvin_models:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export CALVIN_EP_LEN="${MAX_STEPS_PER_TASK}"
    export STARVLA_CALVIN_RENDER_BACKEND
    export STARVLA_CALVIN_RENDER_GPU="${gpu}"
    "${CALVIN_PYTHON}" "${EVAL_PY}" \
      --args.pretrained-path "${CKPT_PATH}" \
      --args.unnorm-key "${UNNORM_KEY}" \
      --args.host 127.0.0.1 \
      --args.port "${port}" \
      --args.dataset-path "${CALVIN_DATASET_PATH}" \
      --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
      --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
      --args.num-ddim-steps "${CALVIN_NUM_DDIM_STEPS}" \
      --args.num-sequences "$((end - start))" \
      --args.max-steps-per-task "${MAX_STEPS_PER_TASK}" \
      --args.sequence-start "${start}" \
      --args.sequence-end "${end}" \
      --args.eval-log-dir "${worker_dir}"
  ) > "${EVAL_ROOT}/eval_${worker}.log" 2>&1 &
  eval_pids+=("$!")
  echo "Started eval worker ${worker}: sequences [${start}, ${end}), GPU ${gpu}, port ${port}, log ${EVAL_ROOT}/eval_${worker}.log"
}

for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((START_GPU + worker / PROCS_PER_GPU))
  port=$((BASE_PORT + worker))
  start=$((worker * NUM_SEQUENCES / NUM_WORKERS))
  end=$(((worker + 1) * NUM_SEQUENCES / NUM_WORKERS))
  server_cmd=(
    "${STAR_VLA_PYTHON}"
    deployment/model_server/server_policy.py
    --ckpt_path "${CKPT_PATH}"
    --port "${port}"
    --idle_timeout -1
    --device "${POLICY_DEVICE}"
    --max_concurrency "${SERVER_CONCURRENCY}"
  )
  if [[ "${USE_BF16}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    server_cmd+=(--use_bf16)
  fi
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=${gpu} "${server_cmd[@]}" \
    > "${EVAL_ROOT}/server_${worker}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "Started policy server ${worker}: GPU ${gpu}, port ${port}, log ${EVAL_ROOT}/server_${worker}_gpu_${gpu}.log"
  wait_for_policy_server "${port}"
  start_eval_worker "${worker}" "${port}" "${start}" "${end}" "${gpu}"
  if (( worker < NUM_WORKERS - 1 )); then
    sleep "${STAGGER_SERVER_SECONDS}"
  fi
done

for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

"${CALVIN_PYTHON}" examples/calvin/eval_files/aggregate_calvin_results.py "${EVAL_ROOT}" | tee "${EVAL_ROOT}/aggregate.log"

cleanup
trap - EXIT
echo "Evaluation complete: ${EVAL_ROOT}"
