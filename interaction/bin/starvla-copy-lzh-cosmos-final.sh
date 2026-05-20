#!/usr/bin/env bash
set -euo pipefail

SRC_RUN="${SRC_RUN:-/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/cosmos_predict2-CosmoPredict2PI-calvin_task_ABC/interactive_calvin_0519_093049}"
SRC_WEIGHT="${SRC_WEIGHT:-${SRC_RUN}/checkpoints/steps_5500_pytorch_model.pt}"
DST_RUN="${DST_RUN:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/LZH/runs/cosmos_predict2-CosmoPredict2PI-calvin_task_ABC_interactive_calvin_0519_093049}"
DST_WEIGHT="${DST_WEIGHT:-${DST_RUN}/checkpoints/steps_5500_pytorch_model.pt}"
DST_FINAL="${DST_FINAL:-${DST_RUN}/final_model/pytorch_model.pt}"

if [[ ! -f "${SRC_WEIGHT}" ]]; then
  echo "[starvla-copy] missing source weight: ${SRC_WEIGHT}" >&2
  exit 1
fi

src_size=$(stat -c '%s' "${SRC_WEIGHT}")
if (( src_size < 10000000000 )); then
  echo "[starvla-copy] source weight is too small (${src_size} bytes); refusing to copy a likely incomplete checkpoint." >&2
  exit 1
fi

mkdir -p "${DST_RUN}/checkpoints" "${DST_RUN}/final_model" "${DST_RUN}/metadata"

copy_one() {
  local src="$1"
  local dst="$2"
  if [[ -f "${dst}" ]]; then
    if cmp -s "${src}" "${dst}"; then
      echo "[starvla-copy] already exists and matches: ${dst}"
      return 0
    fi
    echo "[starvla-copy] destination exists but differs: ${dst}" >&2
    echo "[starvla-copy] remove it manually or set a different DST_RUN." >&2
    exit 1
  fi
  cp -p --reflink=auto "${src}" "${dst}"
}

copy_one "${SRC_WEIGHT}" "${DST_WEIGHT}"
copy_one "${SRC_WEIGHT}" "${DST_FINAL}"

for f in config.yaml config.full.yaml dataset_statistics.json topk_checkpoints.json summary.jsonl; do
  if [[ -f "${SRC_RUN}/${f}" ]]; then
    cp -p "${SRC_RUN}/${f}" "${DST_RUN}/metadata/${f}"
  fi
done

echo "[starvla-copy] copied Cosmos Predict2 PI final weight"
echo "[starvla-copy] source: ${SRC_WEIGHT}"
echo "[starvla-copy] checkpoint copy: ${DST_WEIGHT}"
echo "[starvla-copy] final_model copy: ${DST_FINAL}"
sha256sum "${SRC_WEIGHT}" "${DST_WEIGHT}" "${DST_FINAL}"
find "${DST_RUN}" -maxdepth 2 -type f -printf '%p\t%s\n' | sort
