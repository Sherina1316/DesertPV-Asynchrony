#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/wsy/pv_nature_experiment"; MODE="${1:-full}"
if [[ "$MODE" == test ]]; then OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_test_units_6_28"; LOG_FILE="$PROJECT_ROOT/logs_v24_22_test_units_6_28/test_units_6_28.log"; PID_FILE="$PROJECT_ROOT/pv_v24_22_test_units_6_28.pid"; else OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_novelty_gated_remote_change"; LOG_FILE="$PROJECT_ROOT/logs_v24_22_novelty_gated_remote_change/full_run_v24_22.log"; PID_FILE="$PROJECT_ROOT/pv_v24_22_full.pid"; fi
echo "Mode: $MODE"; echo "Output: $OUTPUT_ROOT"; [[ -f "$PID_FILE" ]] && { pid=$(cat "$PID_FILE"); echo "PID: $pid"; ps -fp "$pid" || true; } || echo "PID file: not found"
echo "Completed stages: $(find "$OUTPUT_ROOT/.step_markers" -maxdepth 1 -type f -name 'step_*_completed.json' 2>/dev/null | wc -l)/21"
find "$OUTPUT_ROOT/.step_markers" -maxdepth 1 -type f -name 'step_*_completed.json' 2>/dev/null | sort || true
echo; echo "Reference and annual-evidence status:"; cat "$OUTPUT_ROOT/07_authoritative_reference/authoritative_reference_status.json" 2>/dev/null || true
echo; echo "Validation status:"; cat "$OUTPUT_ROOT/09_accuracy_validation/accuracy_validation_status.json" 2>/dev/null || true
echo; echo "Recent log:"; tail -100 "$LOG_FILE" 2>/dev/null || true
