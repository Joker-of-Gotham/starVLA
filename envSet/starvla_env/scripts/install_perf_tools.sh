#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="${1:-$(starvla_default_env_dir "${REPO_ROOT}")}"

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"
export STARVLA_REPO_ROOT="${REPO_ROOT}"
export STARVLA_ENVSET_ROOT="${ENVSET_ROOT}"
export STARVLA_ENV_DIR="${ENV_DIR}"

mkdir -p "${ENV_DIR}/bin"
for real in "${ENVSET_ROOT}"/performance_tools/bin/*_perf.real; do
  [ -x "${real}" ] || continue
  name="$(basename "${real%.real}")"
  ln -sfn "${ENVSET_ROOT}/performance_tools/bin/${name}" "${ENV_DIR}/bin/${name}"
done
for tool in p2pBandwidthLatencyTest nvbandwidth; do
  if [ -x "${ENVSET_ROOT}/performance_tools/bin/${tool}" ]; then
    ln -sfn "${ENVSET_ROOT}/performance_tools/bin/${tool}" "${ENV_DIR}/bin/${tool}"
  fi
done

if [ -x "${ENVSET_ROOT}/performance_tools/bin/all_reduce_perf" ]; then
  "${ENVSET_ROOT}/performance_tools/bin/all_reduce_perf" -h >/dev/null
fi

echo "Installed StarVLA performance tool wrappers into ${ENV_DIR}/bin"
