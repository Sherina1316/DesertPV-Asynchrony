#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PYTHON_BIN" -m compileall -q src workflows tools run_all.py
"$PYTHON_BIN" -m pytest -q \
  tests/test_mixed_parcel_recovery_v24_22_2.py \
  tests/test_detail_optimization_v24_22.py \
  tests/test_authoritative_2024_v24_17.py \
  tests/test_adaptive_reference_boundary_v24_18.py \
  tests/test_growth_front_v24_19.py \
  tests/test_anchor_guided.py \
  tests/test_calibration.py \
  tests/test_resource_and_contracts_v24.py \
  tests/test_two_unit_supervision_contract.py
