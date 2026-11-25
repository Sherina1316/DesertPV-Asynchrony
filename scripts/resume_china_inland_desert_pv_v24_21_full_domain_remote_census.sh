#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/wsy/pv_nature_experiment"; CODE_DIR="$PROJECT_ROOT/code/china_inland_desert_pv_v24_22_novelty_gated_remote_change"; PYTHON_BIN="/home/gis007/.venvs/pv_nature/bin/python"
MODE=full; FROM_STEP=1; RERUN=0; BACKGROUND=0
while [[ $# -gt 0 ]]; do case "$1" in --mode) MODE="$2";shift 2;; --from-step) FROM_STEP="$2";shift 2;; --rerun-completed) RERUN=1;shift;; --foreground) BACKGROUND=0;shift;; --background) BACKGROUND=1;shift;; *) echo "Unknown: $1" >&2;exit 2;; esac; done
[[ -f "$CODE_DIR/run_all.py" ]] || { echo "Code not deployed" >&2;exit 1; }
if [[ "$MODE" == test ]]; then OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_test_units_6_28"; LOG_ROOT="$PROJECT_ROOT/logs_v24_22_test_units_6_28"; LOG_FILE="$LOG_ROOT/test_units_6_28.log"; PID_FILE="$PROJECT_ROOT/pv_v24_22_test_units_6_28.pid"; else OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_novelty_gated_remote_change"; LOG_ROOT="$PROJECT_ROOT/logs_v24_22_novelty_gated_remote_change"; LOG_FILE="$LOG_ROOT/full_run_v24_22.log"; PID_FILE="$PROJECT_ROOT/pv_v24_22_full.pid"; fi
CONFIG="$OUTPUT_ROOT/effective_analysis_config.yaml"; [[ -f "$CONFIG" ]] || CONFIG="$CODE_DIR/config/analysis_config.yaml"; mkdir -p "$LOG_ROOT"; export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
CMD=("$PYTHON_BIN" -u "$CODE_DIR/run_all.py" --config "$CONFIG" --from-step "$FROM_STEP" --strict-from-step --allow-scientific-gate-failure); [[ "$RERUN" -eq 1 ]] && CMD+=(--rerun-completed)
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 & echo $! >"$PID_FILE"; echo "PID: $(cat "$PID_FILE")"; else "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; fi
