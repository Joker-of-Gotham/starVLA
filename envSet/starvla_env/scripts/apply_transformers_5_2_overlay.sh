#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

REPO_ROOT="$(starvla_resolve_repo_root "${ENVSET_ROOT}")"
ENV_DIR="${1:-$(starvla_default_env_dir "${REPO_ROOT}")}"
PYTHON="${ENV_DIR}/bin/python"
OVERLAY="${ENVSET_ROOT}/python_overlays/transformers_5_2_stack"

starvla_unset_proxies
starvla_export_offline_paths "${ENVSET_ROOT}"

if [ ! -x "${PYTHON}" ]; then
  echo "Missing python: ${PYTHON}" >&2
  exit 1
fi
if [ ! -d "${OVERLAY}/site-packages" ]; then
  echo "Missing Transformers 5.2 overlay: ${OVERLAY}/site-packages" >&2
  exit 1
fi

SITE_PACKAGES="$("${PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
BACKUP_PARENT="${ENV_DIR}/_starvla_backups"
mkdir -p "${BACKUP_PARENT}"
BACKUP="$(mktemp -d "${BACKUP_PARENT}/transformers_stack_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"

for item in \
  transformers transformers-4.57.0.dist-info transformers-5.2.0.dist-info \
  huggingface_hub huggingface_hub-0.36.2.dist-info huggingface_hub-1.15.0.dist-info \
  typer typer-0.25.1.dist-info typer_slim-0.24.0.dist-info \
  shellingham shellingham-1.5.4.dist-info
do
  if [ -e "${SITE_PACKAGES}/${item}" ]; then
    mv "${SITE_PACKAGES}/${item}" "${BACKUP}/"
  fi
done

for path in "${OVERLAY}/site-packages"/*; do
  [ -e "${path}" ] || continue
  item="$(basename "${path}")"
  rm -rf "${SITE_PACKAGES:?}/${item}"
  cp -a "${path}" "${SITE_PACKAGES}/"
done

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import transformers

assert metadata.version("transformers") == "5.2.0", metadata.version("transformers")
assert hasattr(transformers, "Qwen3_5ForConditionalGeneration")
assert hasattr(transformers, "Qwen3VLForConditionalGeneration")
print("Transformers overlay OK:", transformers.__version__)
PY
