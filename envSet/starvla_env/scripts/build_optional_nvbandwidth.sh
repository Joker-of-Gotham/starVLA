#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVSET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${ENVSET_ROOT}/performance_tools/src/nvbandwidth"
BUILD="${SRC}/build"
OUT="${ENVSET_ROOT}/performance_tools/bin/nvbandwidth"

if [ ! -f "${SRC}/CMakeLists.txt" ]; then
  echo "Missing nvbandwidth source: ${SRC}" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build nvbandwidth." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build nvbandwidth." >&2
  exit 1
fi

cmake -S "${SRC}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j"$(nproc)"
if [ ! -x "${BUILD}/nvbandwidth" ]; then
  echo "nvbandwidth build completed but binary was not found under ${BUILD}." >&2
  exit 1
fi
cp "${BUILD}/nvbandwidth" "${OUT}"
chmod +x "${OUT}"
echo "Installed optional nvbandwidth: ${OUT}"
