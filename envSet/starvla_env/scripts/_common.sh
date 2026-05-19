#!/usr/bin/env bash

starvla_unset_proxies() {
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
}

starvla_is_repo_root() {
  local candidate="$1"
  [ -f "${candidate}/interaction/starvla.py" ] && [ -d "${candidate}/starVLA" ]
}

starvla_resolve_repo_root() {
  local envset_root="$1"
  if [ -n "${STARVLA_REPO_ROOT:-}" ] && starvla_is_repo_root "${STARVLA_REPO_ROOT}"; then
    cd "${STARVLA_REPO_ROOT}" && pwd
    return 0
  fi

  local candidates=(
    "${envset_root}/../.."
    "${envset_root}/../../starVLA"
    "/inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA"
    "$(pwd)"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    candidate="$(cd "${candidate}" 2>/dev/null && pwd || true)"
    if [ -n "${candidate}" ] && starvla_is_repo_root "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done

  echo "Unable to locate StarVLA repo. Set STARVLA_REPO_ROOT=/path/to/starVLA." >&2
  return 1
}

starvla_default_env_dir() {
  local repo_root="$1"
  if [ -n "${STARVLA_ENV_DIR:-}" ]; then
    echo "${STARVLA_ENV_DIR}"
  else
    echo "${repo_root}/env/starvla-py310"
  fi
}

starvla_prepend_path() {
  local path="$1"
  if [ -d "${path}" ]; then
    export PATH="${path}${PATH:+:${PATH}}"
  fi
}

starvla_export_offline_paths() {
  local envset_root="$1"
  starvla_prepend_path "${envset_root}/performance_tools/bin"
  starvla_prepend_path "${envset_root}/system_tools/tmux/bin"
}
