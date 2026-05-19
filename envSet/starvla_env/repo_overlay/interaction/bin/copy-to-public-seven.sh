#!/usr/bin/env bash
set -euo pipefail

SRC="/inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA"
DST="/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starVLA"

mkdir -p "${DST}"

if command -v rsync >/dev/null 2>&1; then
  rsync -aH --numeric-ids --info=progress2 "${SRC}/" "${DST}/"
else
  cp -a "${SRC}/." "${DST}/"
fi

echo "Copied ${SRC} -> ${DST}"
