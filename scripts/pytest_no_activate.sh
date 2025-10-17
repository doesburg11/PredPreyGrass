#!/usr/bin/env bash
# Run pytest using the project-specific conda interpreter without activating the env.
# Keeps your current shell env intact.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY="$PROJECT_ROOT/.conda/bin/python"
cd "$PROJECT_ROOT"
exec "$PY" -m pytest "$@"
