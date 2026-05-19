#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export NUM_GPUS=${NUM_GPUS:-8}
export PROCS_PER_GPU=${PROCS_PER_GPU:-1}
export NUM_WORKERS=${NUM_WORKERS:-$((NUM_GPUS * PROCS_PER_GPU))}
export START_GPU=${START_GPU:-0}
export NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
export BASE_PORT=${BASE_PORT:-5694}

exec "${SCRIPT_DIR}/run_calvin_eval_multigpu_local.sh" "$@"
