#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${SAFENEST_VENV_PATH:-${REPOSITORY_ROOT}/.venv}"

if [[ -f "${REPOSITORY_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPOSITORY_ROOT}/.env"
  set +a
fi

if [[ "${1:-}" == "--install" ]]; then
  shift
  python3 -m venv "${VENV_PATH}"
  "${VENV_PATH}/bin/python" -m pip install --upgrade pip
  "${VENV_PATH}/bin/python" -m pip install \
    -r "${REPOSITORY_ROOT}/requirements-backend.txt" \
    -r "${REPOSITORY_ROOT}/sources/ondevice_ai/requirements-pi.txt"
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "SafeNest virtual environment not found: ${VENV_PATH}" >&2
  echo "Run: bash deployment/run_pi.sh --install" >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"
"${VENV_PATH}/bin/python" -m hil.preflight
exec "${VENV_PATH}/bin/python" backend/run_backend.py "$@"
