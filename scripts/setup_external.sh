#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

SCS_REPO_URL="https://github.com/jostrows9/SCSInSCIMechanisms.git"
SCS_COMMIT="ea349460de2a245ec5d3a929a00006b9ac821825"

clone_or_update() {
  local repo_url="$1"
  local repo_dir="$2"
  local commit="$3"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone "${repo_url}" "${repo_dir}"
  fi

  git -C "${repo_dir}" fetch --all --tags
  git -C "${repo_dir}" checkout "${commit}"
}

install_requirements() {
  local requirements_file="$1"

  if python -m pip --version >/dev/null 2>&1; then
    python -m pip install -r "${requirements_file}"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install -r "${requirements_file}"
    return
  fi

  echo "Neither \`python -m pip\` nor \`uv pip\` is available in the active environment." >&2
  exit 1
}

mkdir -p "${EXTERNAL_DIR}"
clone_or_update "${SCS_REPO_URL}" "${EXTERNAL_DIR}/SCSInSCIMechanisms" "${SCS_COMMIT}"

install_requirements "${EXTERNAL_DIR}/SCSInSCIMechanisms/requirements.txt"

cat <<EOF
External repositories are ready.

Pinned commits:
  SCSInSCIMechanisms: ${SCS_COMMIT}

Next steps:
  1. If needed, install the local package in the active uv environment:
       uv pip install -e ".[optim,dev]"
  2. Run ./scripts/build_neuron.sh
  3. Run python scripts/run_all.py
EOF
