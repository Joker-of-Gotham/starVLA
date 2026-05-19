#!/usr/bin/env bash
set -euo pipefail

OLD_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/seven/seven"
NEW_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/seven/data/starvla"
BACKUP_ROOT="/inspire/qb-ilm2/project/26summer-camp-10/public/seven/seven.migrated.$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -d "${OLD_ROOT}" ]]; then
  echo "Old data root does not exist: ${OLD_ROOT}"
  echo "Nothing to migrate."
  exit 0
fi

mkdir -p "$(dirname "${NEW_ROOT}")"

if [[ ! -e "${NEW_ROOT}" ]]; then
  mv "${OLD_ROOT}" "${NEW_ROOT}"
  echo "Renamed ${OLD_ROOT} -> ${NEW_ROOT}"
else
  echo "New data root already exists: ${NEW_ROOT}"
  echo "Merging contents from ${OLD_ROOT} into ${NEW_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -aH --numeric-ids --info=progress2 "${OLD_ROOT}/" "${NEW_ROOT}/"
  else
    cp -a "${OLD_ROOT}/." "${NEW_ROOT}/"
  fi
  mv "${OLD_ROOT}" "${BACKUP_ROOT}"
  echo "Moved old merged root to backup: ${BACKUP_ROOT}"
fi

if [[ ! -e "${OLD_ROOT}" ]]; then
  ln -s "data/starvla" "${OLD_ROOT}"
  echo "Created compatibility symlink: ${OLD_ROOT} -> data/starvla"
fi

echo "Public StarVLA data root is ready: ${NEW_ROOT}"
