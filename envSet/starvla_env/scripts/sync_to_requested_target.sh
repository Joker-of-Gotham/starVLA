#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${1:-/inspire/qb-ilm2/project/26summer-camp-10/26220447/envSet/starvla_env}"

mkdir -p "${TARGET}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${ENVSET_ROOT}/" "${TARGET}/"
else
  rm -rf "${TARGET:?}/"*
  cp -a "${ENVSET_ROOT}/." "${TARGET}/"
fi

echo "Synced ${ENVSET_ROOT} -> ${TARGET}"
