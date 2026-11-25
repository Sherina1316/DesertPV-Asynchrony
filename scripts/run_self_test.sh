#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gis007/.venvs/pv_nature/bin/python}"
cd "$HERE"
PYTHONPATH="$HERE/src" "$PYTHON_BIN" -m compileall -q src workflows tools run_all.py run_no_leak_experiments.py
PYTHONPATH="$HERE:$HERE/src" "$PYTHON_BIN" -m pytest -q tests
