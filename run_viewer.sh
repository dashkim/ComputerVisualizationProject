#!/usr/bin/env bash
# Run the marching-cube VTK viewer from the repo root (works even if your shell started elsewhere).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
unset VTK_DEFAULT_RENDER_WINDOW_HEADLESS
exec python3 "$ROOT/tools/marching_cube_viewer.py" "$@"
