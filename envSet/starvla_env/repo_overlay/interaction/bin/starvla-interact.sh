#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "bar/activate_starvla_no_proxy.sh" ]]; then
  source "bar/activate_starvla_no_proxy.sh"
fi

python interaction/starvla.py "$@"

