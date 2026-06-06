#!/usr/bin/env bash
set -uo pipefail        # no -e: one crashed seed must not abort the grid

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/experiment_results/experiment5_qcnn/Matrix_$TIMESTAMP"
mkdir -p "$RUN_DIR"

N=8 # adapt to your machine's CPU cores
MON_INTERVAL=30
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""

# Use macOS stat for file mtime
STAT_MTIME_CMD=(stat -f %m)

WORKER="$ROOT_DIR/experiments/exp5_qcnn/run_matrix_one.py"
AGG="$ROOT_DIR/experiments/exp5_qcnn/aggregate.py"

QUADRANTS=("1_HighData_Clean" "2_HighData_Adv" "3_FewShot_Clean" "4_FewShot_Adv")
MODELS=("QCNN" "CNN_Tiny" "CNN_Huge" "QCNN_Hyb")    # add "QCNN_Hyb" for the encoding ablation
SEEDS=({0..9})
CAP=5000 # only useful for big sets
EPOCHS=300

N_TOTAL=$(( ${#QUADRANTS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
START=$(date +%s)
DONE_DIR="$RUN_DIR/.done"; mkdir -p "$DONE_DIR"
trap 'kill "${MON_PID:-}" 2>/dev/null' EXIT

fmt() { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

# Estimate average job duration from done-file mtimes (rolling window)
avg_job_seconds() {
  local done
  done=$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$done" -lt 2 ]; then
    echo 0
    return
  fi
  # collect mtimes
  local mtimes
  mtimes=$(for f in "$DONE_DIR"/*; do ${STAT_MTIME_CMD[@]} "$f" 2>/dev/null; done | sort -n)
  local count
  count=$(echo "$mtimes" | wc -l | tr -d ' ')
  if [ "$count" -lt 2 ]; then
    echo 0
    return
  fi
  # use last up to 6 intervals to smooth spikes
  local tail_count
  tail_count=$(( count > 7 ? 7 : count ))
  local recent
  recent=$(echo "$mtimes" | tail -n "$tail_count")
  local prev=""
  local sum=0
  local n=0
  for t in $recent; do
    if [ -n "$prev" ]; then
      local diff=$(( t - prev ))
      if [ "$diff" -gt 0 ]; then
        sum=$(( sum + diff ))
        n=$(( n + 1 ))
      fi
    fi
    prev=$t
  done
  if [ "$n" -eq 0 ]; then
    echo 0
  else
    echo $(( sum / n ))
  fi
}

monitor() {
  while true; do
    local done now el avg eta
    done=$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l | tr -d ' ')
    now=$(date +%s); el=$(( now - START ))
    avg=$(avg_job_seconds)

    if [ "$done" -gt 0 ] && [ "$avg" -gt 0 ]; then
      # Remaining jobs / concurrency -> remaining batches * avg duration
      local remaining=$(( N_TOTAL - done ))
      local effective=$(( remaining / (N>0?N:1) ))
      # ceil division for batches
      if [ $(( remaining % (N>0?N:1) )) -ne 0 ]; then
        effective=$(( effective + 1 ))
      fi
      eta=$(( effective * avg ))
      printf "[PROGRESS] %s | done %d/%d | elapsed %s | avg/job ~%ds | ETA ~%s\n" \
        "$(date +%H:%M:%S)" "$done" "$N_TOTAL" "$(fmt $el)" "$avg" "$(fmt $eta)"
    else
      printf "[PROGRESS] %s | done %d/%d | elapsed %s | warming up...\n" \
        "$(date +%H:%M:%S)" "$done" "$N_TOTAL" "$(fmt $el)"
    fi
    [ "$done" -ge "$N_TOTAL" ] && break
    sleep "$MON_INTERVAL"
  done
}

echo "Run dir:    $RUN_DIR"
echo "Total jobs: $N_TOTAL  (max $N concurrent)"
echo "Started:    $(date)"
monitor & MON_PID=$!; disown "$MON_PID"

for Q in "${QUADRANTS[@]}"; do
  for M in "${MODELS[@]}"; do
    for S in "${SEEDS[@]}"; do
      TAG="${Q}_${M}_seed${S}"
      LOG_FILE="$RUN_DIR/${TAG}.log"
      (
        trap - EXIT INT TERM
        echo "=== $TAG | started $(date) ===" > "$LOG_FILE"
        python "$WORKER" --quadrant "$Q" --model "$M" --seed "$S" \
               --run-dir "$RUN_DIR" --cap-per-class "$CAP" --epochs "$EPOCHS" \
               >> "$LOG_FILE" 2>&1
        echo "=== $TAG | finished (exit $?) at $(date) ===" >> "$LOG_FILE"
        touch "$DONE_DIR/$TAG"            # mark complete regardless of exit code
      ) &
      while [ "$(jobs -r | wc -l)" -ge "$N" ]; do wait -n || true; done
    done
  done
done
wait || true

echo "All jobs done at $(date). Total wall time: $(fmt $(( $(date +%s) - START )))"
echo "Aggregating..."
python "$AGG" --run-dir "$RUN_DIR" --baselines CNN_Tiny CNN_Huge
echo "Done. Results + plots under: $RUN_DIR"
