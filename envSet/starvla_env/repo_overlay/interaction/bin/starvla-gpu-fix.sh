#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

candidates=(
  "${REPO_ROOT}/envSet/starvla_env"
  "/inspire/qb-ilm2/project/26summer-camp-10/26220447/envSet/starvla_env"
  "/inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA/envSet/starvla_env"
)

for envset_root in "${candidates[@]}"; do
  script="${envset_root}/scripts/gpu_oneclick_fix.sh"
  if [ -x "${script}" ]; then
    exec bash "${script}" --repo-root "${REPO_ROOT}" "$@"
  fi
done

echo "Unable to locate envSet/starvla_env/scripts/gpu_oneclick_fix.sh." >&2
echo "Expected one of:" >&2
for envset_root in "${candidates[@]}"; do
  echo "  ${envset_root}" >&2
done
exit 1
