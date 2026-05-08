#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-plot_disjoint}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-${INVITA_DATA_ROOT:-../INVITA-WheatFieldState-derived_release}}"
SPLIT_ROOT="${SPLIT_ROOT:-${INVITA_SPLIT_ROOT:-${DATA_ROOT}/splits/plot_disjoint}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INVITA_OUTPUT_ROOT:-outputs}}"
NEURAL_ROOT="${NEURAL_ROOT:-${OUTPUT_ROOT}/neural_baselines}"
LOG_DIR="${LOG_DIR:-${NEURAL_ROOT}/logs_${RUN_ID}}"
mkdir -p "$LOG_DIR"

PROGRESS_CSV="$LOG_DIR/progress.csv"
PROGRESS_MD="$LOG_DIR/progress.md"
MASTER_LOG="$LOG_DIR/run_neural_representations.log"

printf "step,status,start_time,end_time,elapsed_seconds,log_path\n" > "$PROGRESS_CSV"
cat > "$PROGRESS_MD" <<EOF
# Neural Representation Progress

- run_id: \`$RUN_ID\`
- device: \`$DEVICE\`
- data_root: \`$DATA_ROOT\`
- split_root: \`$SPLIT_ROOT\`
- log_dir: \`$LOG_DIR\`

| Step | Status | Start | End | Elapsed | Log |
| --- | --- | --- | --- | ---: | --- |
EOF

log_msg() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$MASTER_LOG"
}

append_progress() {
  local step="$1"
  local status="$2"
  local start_time="$3"
  local end_time="$4"
  local elapsed="$5"
  local log_path="$6"

  printf '%s,%s,%s,%s,%s,%s\n' \
    "$step" "$status" "$start_time" "$end_time" "$elapsed" "$log_path" >> "$PROGRESS_CSV"
  printf '| %s | %s | %s | %s | %s | `%s` |\n' \
    "$step" "$status" "$start_time" "$end_time" "$elapsed" "$log_path" >> "$PROGRESS_MD"
}

run_step() {
  local step="$1"
  shift
  local log_path="$LOG_DIR/${step}.log"
  local start_time
  local end_time
  local start_epoch
  local end_epoch
  local elapsed
  local status

  start_time="$(date -Iseconds)"
  start_epoch="$(date +%s)"
  log_msg "START $step"
  log_msg "LOG $log_path"

  set +e
  "$@" 2>&1 | tee "$log_path"
  status="${PIPESTATUS[0]}"
  set -e

  end_time="$(date -Iseconds)"
  end_epoch="$(date +%s)"
  elapsed="$((end_epoch - start_epoch))"

  if [[ "$status" -eq 0 ]]; then
    log_msg "DONE  $step elapsed=${elapsed}s"
    append_progress "$step" "done" "$start_time" "$end_time" "$elapsed" "$log_path"
  else
    log_msg "FAIL  $step status=$status elapsed=${elapsed}s"
    append_progress "$step" "failed:$status" "$start_time" "$end_time" "$elapsed" "$log_path"
    exit "$status"
  fi
}

log_msg "Progress CSV: $PROGRESS_CSV"
log_msg "Progress MD:  $PROGRESS_MD"

COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --split-root "$SPLIT_ROOT"
  --run-id "$RUN_ID"
  --device "$DEVICE"
  --seed 42
)

run_step tabular_transformer \
  "$PYTHON" experiments/runs/run_tabular_transformer.py \
  "${COMMON_ARGS[@]}"

run_step observation_set_transformer \
  "$PYTHON" experiments/runs/run_observation_set_transformer.py \
  "${COMMON_ARGS[@]}" \
  --max-tokens 64

run_step sensor_sequence_tcn \
  "$PYTHON" experiments/runs/run_sensor_sequence_tcn.py \
  "${COMMON_ARGS[@]}" \
  --model tcn

run_step field_camera_image_set_model \
  "$PYTHON" experiments/runs/run_field_camera_image_set_model.py \
  "${COMMON_ARGS[@]}" \
  --encoder frozen_image_feature_squeezenet_fallback

run_step gated_stacker \
  "$PYTHON" experiments/runs/run_gated_stacker.py \
  --results-root "$OUTPUT_ROOT" \
  --neural-results-root "$NEURAL_ROOT" \
  --output-dir "$NEURAL_ROOT/gated_stacker" \
  --run-id "$RUN_ID" \
  --routes tabular_metadata,observation_availability,sensor_summary,frozen_image_feature,sensor_sequence_transformer,tabular_transformer,observation_set_transformer,sensor_sequence_tcn,field_camera_image_set_model \
  --route-root "tabular_transformer=$NEURAL_ROOT/tabular_transformer/$RUN_ID" \
  --route-root "observation_set_transformer=$NEURAL_ROOT/observation_set_transformer/$RUN_ID" \
  --route-root "sensor_sequence_tcn=$NEURAL_ROOT/sensor_sequence_tcn/$RUN_ID" \
  --route-root "field_camera_image_set_model=$NEURAL_ROOT/field_camera_image_set_model/$RUN_ID" \
  --stacking-policy validation_only \
  --device "$DEVICE" \
  --seed 42

run_step summarize_neural_representations \
  "$PYTHON" experiments/runs/summarize_neural_representations.py \
  --results-root "$OUTPUT_ROOT" \
  --neural-results-root "$NEURAL_ROOT" \
  --output-dir "$NEURAL_ROOT/summary_${RUN_ID}" \
  --run-id "$RUN_ID"

log_msg "All neural representation steps completed."
