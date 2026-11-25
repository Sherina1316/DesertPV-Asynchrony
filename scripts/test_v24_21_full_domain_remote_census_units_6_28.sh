#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/wsy/pv_nature_experiment"
ZIP_PATH="/home/wsy/china_inland_desert_pv_v24_22_novelty_gated_remote_change.zip"
CODE_PARENT="$PROJECT_ROOT/code"
CODE_DIR="$CODE_PARENT/china_inland_desert_pv_v24_22_novelty_gated_remote_change"
PYTHON_BIN="/home/gis007/.venvs/pv_nature/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/output_v24_22_test_units_6_28"
CACHE_ROOT="$PROJECT_ROOT/cache_v24_22_test_units_6_28"
LOG_ROOT="$PROJECT_ROOT/logs_v24_22_test_units_6_28"
MODEL_ROOT="$PROJECT_ROOT/models_v24_22_test_units_6_28"
LOG_FILE="$LOG_ROOT/test_units_6_28.log"
PID_FILE="$PROJECT_ROOT/pv_v24_22_test_units_6_28.pid"
CONFIG="$CODE_DIR/config/analysis_config.yaml"
YES=0; BACKGROUND=0; SELF_TEST=1; FROM_STEP=1; RERUN=0; SUMMARY_ONLY=0
while [[ $# -gt 0 ]]; do
 case "$1" in
 --yes) YES=1;shift;; --foreground) BACKGROUND=0;shift;; --background) BACKGROUND=1;shift;; --no-self-test) SELF_TEST=0;shift;; --from-step) FROM_STEP="$2";shift 2;; --rerun-completed) RERUN=1;shift;; --summary-only) SUMMARY_ONLY=1;shift;; *) echo "Unknown option: $1" >&2;exit 2;; esac
done
if [[ "$SUMMARY_ONLY" -eq 1 ]]; then "$PYTHON_BIN" "$CODE_DIR/tools/summarize_reference_test.py" --output-root "$OUTPUT_ROOT" --unit-ids 6,28; exit $?; fi
cat <<EOF
v24.22 full-domain remote-census reconstruction diagnostic run: Units 6 and 28 only
Reference: /home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp
Contracts:
  - trusted manual interior is retained;
  - only a 120 m boundary-uncertainty band can be changed;
  - additions require multiband installation break and PV-state evidence;
  - removals require strong non-PV/sand evidence and weak change support;
  - the 2024 endpoint uses only 2011-2024 evidence; 2025 is evaluated after the endpoint is frozen;
  - environment cannot create a historical PV label.
Output: $OUTPUT_ROOT
EOF
[[ "$YES" -eq 1 ]] || { read -r -p "Continue? [y/N] " a; [[ "$a" =~ ^[Yy]$ ]] || exit 0; }
[[ -f "$ZIP_PATH" ]] || { echo "Missing ZIP" >&2; exit 1; }; for ext in shp shx dbf prj; do
  [[ -f "/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.${ext}" ]] || { echo "Missing corrected SHP sidecar: pv_desert_fix.${ext}" >&2; exit 1; }
done
if [[ -f "$PID_FILE" ]]; then kill "$(cat "$PID_FILE")" 2>/dev/null || true; fi
pkill -f "$CODE_DIR/run_all.py" 2>/dev/null || true; sleep 1; rm -f "$PID_FILE" "$PROJECT_ROOT"/*v24_22_test*.lock; mkdir -p "$CODE_PARENT"
if [[ "$FROM_STEP" -eq 1 ]]; then rm -rf "$CODE_DIR"; unzip -q "$ZIP_PATH" -d "$CODE_PARENT"; else [[ -f "$CODE_DIR/run_all.py" ]] || { rm -rf "$CODE_DIR"; unzip -q "$ZIP_PATH" -d "$CODE_PARENT"; }; fi
export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
"$PYTHON_BIN" - <<PY2
from pathlib import Path
import yaml
p=Path('$CONFIG'); cfg=yaml.safe_load(p.read_text()); ids=[6,28]
cfg['study']['selected_unit_ids']=ids
for k,v in {'output_root':'$OUTPUT_ROOT','cache_root':'$CACHE_ROOT','log_root':'$LOG_ROOT','model_root':'$MODEL_ROOT'}.items(): cfg['paths'][k]=v
cfg['paths']['reference_shp']='/home/wsy/pv_nature_experiment/input_tables/pv_desert_fix.shp'
cfg['processing']['run_mode']='v24_22_test_units_6_28_novelty_gated_remote_change'
cfg.setdefault('test_mode',{})['enabled_when_selected_unit_ids']=True; cfg['test_mode']['relax_formal_production_contracts']=True; cfg['test_mode']['minimum_positive_regions']=2
cfg['supervision']['minimum_positive_regions']=2
sp=cfg['supervision']['split']; sp.update({'minimum_positive_train_units':1,'minimum_positive_calibration_units':1,'minimum_positive_test_units':0,'train_fraction':0.5,'calibration_fraction':0.5,'test_fraction':0.0})
cfg['patches']['minimum_training_regions']=1; cfg['patches']['spatial_folds']=2; cfg['training']['minimum_valid_spatial_folds']=1; cfg['training']['minimum_positive_validation_pixels']=1; cfg['training']['minimum_negative_validation_pixels']=1
cfg['validation']['minimum_test_positive_units']=0; cfg['quality_routing']['minimum_direct_units']=1
cfg['authoritative_reference']['diagnostic_unit_ids']=ids; cfg['authoritative_reference']['write_diagnostics']=True
cfg['reference_reconstruction']=cfg['authoritative_reference'].copy(); cfg['reference_temporal_reconstruction']=cfg['authoritative_temporal_reconstruction'].copy()
cfg['future_2050']['require_scientific_gate']=False; cfg['ecology']['require_scientific_gate_for_formal_results']=False
p.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)); print('Selected units:',ids)
PY2
if [[ "$SELF_TEST" -eq 1 && "$FROM_STEP" -eq 1 ]]; then (cd "$CODE_DIR"; PYTHONPATH=src "$PYTHON_BIN" -m compileall -q src workflows tools run_all.py; PYTHONPATH=src:. "$PYTHON_BIN" -m pytest -q tests/test_authoritative_2024_v24_17.py tests/test_adaptive_reference_boundary_v24_18.py tests/test_growth_front_v24_19.py tests/test_causal_terminal_extension_v24_22.py tests/test_novelty_gated_remote_change_v24_22.py tests/test_supervision_assignment.py tests/test_calibration.py); fi
if [[ "$FROM_STEP" -eq 1 ]]; then rm -rf "$OUTPUT_ROOT" "$CACHE_ROOT" "$LOG_ROOT" "$MODEL_ROOT"; fi; mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT" "$LOG_ROOT" "$MODEL_ROOT"
CMD=("$PYTHON_BIN" -u "$CODE_DIR/run_all.py" --config "$CONFIG" --from-step "$FROM_STEP" --allow-scientific-gate-failure); [[ "$FROM_STEP" -gt 1 ]] && CMD+=(--strict-from-step); [[ "$RERUN" -eq 1 ]] && CMD+=(--rerun-completed)
run(){ set +e; "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; rc=${PIPESTATUS[0]}; set -e; "$PYTHON_BIN" "$CODE_DIR/tools/summarize_reference_test.py" --output-root "$OUTPUT_ROOT" --unit-ids 6,28 2>&1 | tee -a "$LOG_FILE" || true; return "$rc"; }
if [[ "$BACKGROUND" -eq 1 ]]; then nohup bash -c "$(declare -f run); $(declare -p CMD PYTHON_BIN CODE_DIR OUTPUT_ROOT LOG_FILE); run" >"$LOG_FILE" 2>&1 & echo $! >"$PID_FILE"; echo "PID: $(cat "$PID_FILE")"; else run; fi
