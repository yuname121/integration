#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${INTEGRATION_ROOT}/.." && pwd)"
VENV_PATH="${SAFENEST_VENV_PATH:-${REPOSITORY_ROOT}/.venv}"

if [[ "${1:-}" == "--install" ]]; then
  shift
  python3 -m venv "${VENV_PATH}"
  "${VENV_PATH}/bin/python" -m pip install --upgrade pip
  "${VENV_PATH}/bin/python" -m pip install \
    -r "${INTEGRATION_ROOT}/requirements-backend.txt" \
    -r "${INTEGRATION_ROOT}/sources/ondevice_ai/requirements-pi.txt"
fi

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "SafeNest virtual environment not found: ${VENV_PATH}" >&2
  echo "Run: bash deployment/run_pi.sh --install" >&2
  exit 2
fi

cd "${REPOSITORY_ROOT}"
"${VENV_PATH}/bin/python" -m hil.preflight
exec "${VENV_PATH}/bin/python" backend/run_backend.py "$@"
