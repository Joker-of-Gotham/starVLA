#!/usr/bin/env bash
# Source this file: source scripts/activate_offline_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")" || return 1 2>/dev/null || exit 1
ENV_DIR="$(starvla_default_env_dir "${REPO_ROOT}")"

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"

if [ -f "${ENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"
else
  echo "Missing StarVLA env: ${ENV_DIR}" >&2
  return 1 2>/dev/null || exit 1
fi

export STARVLA_REPO_ROOT="${REPO_ROOT}"
export STARVLA_ENVSET_ROOT="${ENVSET_ROOT}"
export STARVLA_ENV_DIR="${ENV_DIR}"
