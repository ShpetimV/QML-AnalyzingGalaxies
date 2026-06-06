#!/usr/bin/env bash
set -uo pipefail        # no -e: one crashed seed must not abort the grid

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/experiment_results/experiment4_quanv_noise_robustness/$TIMESTAMP"
mkdir -p "$RUN_DIR"

N=7                     # 7 workers + ~1 core headroom for OS/monitor/shell
MON_INTERVAL=30         # seconds between progress prints
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""

HEADS=("heavy" "light")
SEEDS=({0..0})
EXPERIMENTS=(
  "experiments/exp4_noise/run_quanv_ood.py ood"
  "experiments/exp4_noise/run_quanv_adv.py adv"
)

N_TOTAL=$(( ${#EXPERIMENTS[@]} * ${#HEADS[@]} * ${#SEEDS[@]} ))
START=$(date +%s)
DONE_DIR="$(mktemp -d)"
trap 'kill "${MON_PID:-}" 2>/dev/null; rm -rf "$DONE_DIR"' EXIT

fmt() { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

monitor() {
  while true; do
    local done now el avg eta
    done=$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l)
    now=$(date +%s); el=$(( now - START ))
    if [ "$done" -gt 0 ]; then
      avg=$(( el / done ))
      eta=$(( (N_TOTAL - done) * avg ))
      printf "[PROGRESS] %s | done %d/%d | elapsed %s | avg/job %ds | ETA %s\n" \
        "$(date +%H:%M:%S)" "$done" "$N_TOTAL" "$(fmt $el)" "$avg" "$(fmt $eta)"
    else
      printf "[PROGRESS] %s | done 0/%d | elapsed %s | warming up...\n" \
        "$(date +%H:%M:%S)" "$N_TOTAL" "$(fmt $el)"
    fi
    [ "$done" -ge "$N_TOTAL" ] && break
    sleep "$MON_INTERVAL"
  done
}

echo "Run dir:   $RUN_DIR"
echo "Total jobs: $N_TOTAL  (max $N concurrent)"
echo "Started:   $(date)"

monitor & MON_PID=$!; disown "$MON_PID"      # disown so it isn't counted by `jobs -r`

for exp in "${EXPERIMENTS[@]}"; do
  IFS=" " read -r SCRIPT EXP_NAME <<< "$exp"
  for HEAD in "${HEADS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      LOG_FILE="$RUN_DIR/${EXP_NAME}_${HEAD}_seed${SEED}.log"
      TAG="${EXP_NAME}_${HEAD}_seed${SEED}"
      (
        echo "=== $TAG | started $(date) ===" > "$LOG_FILE"
        uv run python "$SCRIPT" --head "$HEAD" --seeds "$SEED" --run-dir "$RUN_DIR" >> "$LOG_FILE" 2>&1
        echo "=== $TAG | finished (exit $?) at $(date) ===" >> "$LOG_FILE"
        touch "$DONE_DIR/$TAG"                # mark complete regardless of exit code
      ) &
      while [ "$(jobs -r | wc -l)" -ge "$N" ]; do wait -n || true; done
    done
  done
done
wait || true

echo "All jobs done at $(date). Total wall time: $(fmt $(( $(date +%s) - START )))"
echo "Aggregating..."
uv run python "$ROOT_DIR/experiments/exp4_noise/aggregate.py" --run-dir "$RUN_DIR"
echo "Done. Logs + results + plots under: $RUN_DIR"