#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/env/starvla-py310/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

RUN_ID="${RUN_ID:-calvin_ultimate_moe_soup_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/ensembles/${RUN_ID}}"
METHOD="${METHOD:-weighted_soup}"

cd "${REPO_ROOT}"
exec "${PYTHON}" interaction/starvla.py ensemble \
  --recipe calvin_ultimate_moe \
  --method "${METHOD}" \
  --output-dir "${OUTPUT_DIR}" \
  --reference-index "${REFERENCE_INDEX:-0}" \
  --overwrite \
  --yes
