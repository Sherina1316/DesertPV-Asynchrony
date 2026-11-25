#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/wsy/pv_nature_experiment"
ZIP_PATH="/home/wsy/china_inland_desert_pv_v24_22_novelty_gated_remote_change.zip"
CODE_PARENT="$PROJECT_ROOT/code"
CODE_DIR="$CODE_PARENT/china_inland_desert_pv_v24_22_novelty_gated_remote_change"
PYTHON_BIN="/home/gis007/.venvs/pv_nature/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_novelty_gated_remote_change"
CACHE_ROOT="$PROJECT_ROOT/cache_v24_22_novelty_gated_remote_change"
LOG_ROOT="$PROJECT_ROOT/logs_v24_22_novelty_gated_remote_change"
MODEL_ROOT="$PROJECT_ROOT/models_v24_22_novelty_gated_remote_change"
LOG_FILE="$LOG_ROOT/full_run_v24_22.log"
PID_FILE="$PROJECT_ROOT/pv_v24_22_full.pid"
CONFIG="$CODE_DIR/config/analysis_config.yaml"
YES=0; BACKGROUND=0; SELF_TEST=1
while [[ $# -gt 0 ]]; do
 case "$1" in
  --yes) YES=1; shift;; --foreground) BACKGROUND=0; shift;; --background) BACKGROUND=1; shift;; --no-self-test) SELF_TEST=0; shift;; *) echo "Unknown option: $1" >&2; exit 2;; esac
done
cat <<EOF
v24.22 full 142-unit full-domain remote-census reconstruction run
Reference: /home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp
2024 uses the manual vector as a trusted prior; the 2024 boundary is refined conservatively; pre-2024 extents are completed by connected growth fronts; 2025 additions use gain-supported, remote-site and component-conditioned perimeter pathways after a causal 2024 freeze.
Code: $CODE_DIR
Output: $OUTPUT_ROOT
EOF
[[ "$YES" -eq 1 ]] || { read -r -p "Continue? [y/N] " a; [[ "$a" =~ ^[Yy]$ ]] || exit 0; }
[[ -f "$ZIP_PATH" ]] || { echo "Missing ZIP: $ZIP_PATH" >&2; exit 1; }
for ext in shp shx dbf prj; do
  [[ -f "/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.${ext}" ]] || { echo "Missing corrected SHP sidecar: pv_desert_fix.${ext}" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
pkill -f "$CODE_DIR/run_all.py" 2>/dev/null || true; sleep 1
rm -f "$PID_FILE" "$PROJECT_ROOT"/*v24_22*.lock
rm -rf "$CODE_DIR"; mkdir -p "$CODE_PARENT"; unzip -q "$ZIP_PATH" -d "$CODE_PARENT"
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
"$PYTHON_BIN" - <<PY2
from pathlib import Path
import yaml
p=Path('$CONFIG'); cfg=yaml.safe_load(p.read_text())
cfg['study']['selected_unit_ids']=[]
for k,v in {'output_root':'$OUTPUT_ROOT','cache_root':'$CACHE_ROOT','log_root':'$LOG_ROOT','model_root':'$MODEL_ROOT'}.items(): cfg['paths'][k]=v
cfg['paths']['reference_shp']='/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp'
cfg['processing']['run_mode']='v24_22_full_142_novelty_gated_remote_change'
cfg['authoritative_reference']['write_diagnostics']=False
p.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
print('Configured full 142-unit run; reference:',cfg['paths']['reference_shp'])
PY2
if [[ "$SELF_TEST" -eq 1 ]]; then (cd "$CODE_DIR"; PYTHONPATH=src "$PYTHON_BIN" -m compileall -q src workflows tools run_all.py; PYTHONPATH=src:. "$PYTHON_BIN" -m pytest -q tests/test_authoritative_2024_v24_17.py tests/test_adaptive_reference_boundary_v24_18.py tests/test_growth_front_v24_19.py tests/test_causal_terminal_extension_v24_22.py tests/test_novelty_gated_remote_change_v24_22.py tests/test_supervision_assignment.py tests/test_calibration.py tests/test_two_part_growth_forecast_v24_13.py); fi
rm -rf "$OUTPUT_ROOT" "$CACHE_ROOT" "$LOG_ROOT" "$MODEL_ROOT"; mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$LOG_ROOT" "$MODEL_ROOT"
CMD=("$PYTHON_BIN" -u "$CODE_DIR/run_all.py" --config "$CONFIG" --from-step 1 --strict-from-step --allow-scientific-gate-failure)
if [[ "$BACKGROUND" -eq 1 ]]; then nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 & echo $! >"$PID_FILE"; echo "PID: $(cat "$PID_FILE")"; else "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; fi
