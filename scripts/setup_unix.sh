#!/usr/bin/env sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Python 3.11 was not found." >&2
  exit 1
fi

"$PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10-3.12 is required"'
"$PYTHON" -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/scripts/download_models.py"

echo "Setup complete. Start with: ./.venv/bin/python eye_server.py"
