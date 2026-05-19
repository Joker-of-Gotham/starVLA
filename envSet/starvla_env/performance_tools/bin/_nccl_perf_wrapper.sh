#!/usr/bin/env bash
set -euo pipefail

TOOL_NAME="$(basename "$0")"
SCRIPT_PATH="${BASH_SOURCE[0]}"
while [ -L "${SCRIPT_PATH}" ]; do
  LINK_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
  LINK_TARGET="$(readlink "${SCRIPT_PATH}")"
  case "${LINK_TARGET}" in
    /*) SCRIPT_PATH="${LINK_TARGET}" ;;
    *) SCRIPT_PATH="${LINK_DIR}/${LINK_TARGET}" ;;
  esac
done
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REAL="${SCRIPT_DIR}/${TOOL_NAME}.real"

if [ ! -x "${REAL}" ]; then
  echo "Missing packaged NCCL test binary: ${REAL}" >&2
  exit 127
fi

ENV_DIR="${STARVLA_ENV_DIR:-}"
if [ -z "${ENV_DIR}" ]; then
  for candidate in \
    "${ENVSET_ROOT}/../../env/starvla-py310" \
    "${ENVSET_ROOT}/../../starVLA/env/starvla-py310" \
    "/inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA/env/starvla-py310"; do
    if [ -d "${candidate}" ]; then
      ENV_DIR="$(cd "${candidate}" && pwd)"
      break
    fi
  done
fi

SITE_PACKAGES="${ENV_DIR}/lib/python3.10/site-packages"
LIBS=(
  "${ENVSET_ROOT}/performance_tools/lib/verifiable"
  "${SITE_PACKAGES}/nvidia/nccl/lib"
  "${SITE_PACKAGES}/nvidia/cuda_runtime/lib"
  "/usr/local/cuda/targets/x86_64-linux/lib"
)

for path in "${LIBS[@]}"; do
  if [ -d "${path}" ]; then
    export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done

exec "${REAL}" "$@"
