#!/usr/bin/env bash
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

RUN_DIR="/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/20260519_173434_interactive_calvin_0519_163215"
WATCH_PATH="/inspire/qb-ilm2/project/26summer-camp-10/public"
OUTPUT_DIR=""
RUNNER=""
STOP_FREE_GB="auto"
RESUME_FREE_GB="auto"
POLL_SECONDS=60
GRACE_SECONDS=90
MAX_RESTARTS=0
STATUS_ONLY=0
DRY_RUN=0
ABORT_ON_CUDA_OOM=0
GIB=$((1024 * 1024 * 1024))

usage() {
  cat <<'USAGE'
usage: bash interaction/bin/starvla-disk-guard-resume.sh [options]

  --run-dir PATH
  --checkpoint-output-dir PATH   Defaults to job.json output_dir.
  --watch-path PATH              Default: /inspire/.../public
  --runner PATH                  Default: RUN_DIR/runners/main.runner.sh
  --stop-free-gb N|auto          Low watermark; stop below it.
  --resume-free-gb N|auto        High watermark; resume at/above it.
  --poll-seconds N               Default: 60
  --grace-seconds N              Default: 90
  --max-restarts N               0 means unlimited.
  --status                       Print status and exit.
  --dry-run
  --abort-on-cuda-oom          Stop instead of retrying if main.log contains CUDA OOM.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --checkpoint-output-dir|--checkpoint-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --watch-path) WATCH_PATH="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --stop-free-gb) STOP_FREE_GB="$2"; shift 2 ;;
    --resume-free-gb) RESUME_FREE_GB="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --grace-seconds) GRACE_SECONDS="$2"; shift 2 ;;
    --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
    --status) STATUS_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --abort-on-cuda-oom) ABORT_ON_CUDA_OOM=1; shift ;;
    --no-abort-on-cuda-oom) ABORT_ON_CUDA_OOM=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$RUN_DIR" ]] || { echo "[disk-guard] run dir not found: $RUN_DIR" >&2; exit 2; }
JOB_JSON="$RUN_DIR/job.json"
if [[ -z "$OUTPUT_DIR" && -f "$JOB_JSON" ]]; then
  OUTPUT_DIR="$(python - "$JOB_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("output_dir", ""))
PY
)"
fi
[[ -n "$OUTPUT_DIR" ]] || { echo "[disk-guard] missing --checkpoint-output-dir" >&2; exit 2; }

RUNNER="${RUNNER:-$RUN_DIR/runners/main.runner.sh}"
GUARD_DIR="$RUN_DIR"
if ! (mkdir -p "$GUARD_DIR" 2>/dev/null && touch "$GUARD_DIR/.disk_guard_write_test" 2>/dev/null); then
  GUARD_DIR="/tmp/starvla_disk_guard_$(basename "$RUN_DIR")"
  mkdir -p "$GUARD_DIR"
else
  rm -f "$GUARD_DIR/.disk_guard_write_test" 2>/dev/null || true
fi
LOG_FILE="$GUARD_DIR/disk_guard.log"
EVENTS_FILE="$GUARD_DIR/disk_guard.events.jsonl"
PID_FILE="$GUARD_DIR/disk_guard.pid"
MAIN_LOG="$RUN_DIR/logs/main.log"
STATE_ROOT="$OUTPUT_DIR/accelerate_state"
WEIGHT_DIR="$OUTPUT_DIR/checkpoints"
FINAL_DIR="$OUTPUT_DIR/final_model"
mkdir -p "$(dirname "$LOG_FILE")"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { local m="[$(ts)] $*"; echo "$m"; printf '%s\n' "$m" >> "$LOG_FILE" || true; }
event() { printf '{"time":"%s","event":"%s","detail":"%s"}\n' "$(ts)" "$1" "${2:-}" >> "$EVENTS_FILE" || true; }
free_bytes() { df -PB1 "$WATCH_PATH" | awk 'NR==2 {print $4}'; }
gb_to_bytes() { python - "$1" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024**3))
PY
}
human() { command -v numfmt >/dev/null 2>&1 && numfmt --to=iec-i --suffix=B "$1" || python - "$1" <<'PY'
import sys
n=float(sys.argv[1])
for u in ["B","KiB","MiB","GiB","TiB"]:
    if n < 1024 or u == "TiB": print(f"{n:.2f} {u}"); break
    n /= 1024
PY
}
du_bytes() { [[ -e "$1" ]] && du -sb "$1" 2>/dev/null | awk 'NR==1 {print $1+0}' || echo 0; }
file_size() { stat -c '%s' "$1" 2>/dev/null || echo 0; }
step_from_name() { sed -nE 's/.*steps_([0-9]+).*/\1/p' <<<"$1"; }

latest_summary_step() {
  local summary="$OUTPUT_DIR/summary.jsonl"
  [[ -f "$summary" ]] || { echo 0; return; }
  python - "$summary" <<'PY'
import json, sys
best = 0
for line in open(sys.argv[1], encoding="utf-8"):
    try: best = max(best, int(json.loads(line).get("steps") or 0))
    except Exception: pass
print(best)
PY
}
latest_state_step() {
  if [[ -e "$STATE_ROOT/latest" ]]; then step_from_name "$(readlink -f "$STATE_ROOT/latest" 2>/dev/null || true)"; return; fi
  find "$STATE_ROOT" -maxdepth 1 -type d -name 'steps_*' -printf '%f\n' 2>/dev/null | sed -nE 's/.*steps_([0-9]+).*/\1/p' | sort -n | tail -1
}
latest_valid_weight_step() {
  local min_bytes; min_bytes="$(gb_to_bytes 1)"
  find "$WEIGHT_DIR" -maxdepth 1 -type f \( -name 'steps_*_pytorch_model.pt' -o -name 'steps_*_model.safetensors' \) -print 2>/dev/null |
    while read -r f; do [[ "$(file_size "$f")" -ge "$min_bytes" ]] && step_from_name "$f"; done | sort -n | tail -1
}
estimate_budget_bytes() {
  local state_max=0 weight_max=0 min_weight size fallback budget
  min_weight="$(gb_to_bytes 1)"
  while read -r d; do [[ -n "$d" ]] || continue; size="$(du_bytes "$d")"; [[ "$size" -gt "$state_max" ]] && state_max="$size"; done < <(find "$STATE_ROOT" -maxdepth 1 -type d -name 'steps_*' -print 2>/dev/null)
  while read -r f; do [[ -n "$f" ]] || continue; size="$(file_size "$f")"; [[ "$size" -ge "$min_weight" && "$size" -gt "$weight_max" ]] && weight_max="$size"; done < <(find "$WEIGHT_DIR" -maxdepth 1 -type f \( -name 'steps_*_pytorch_model.pt' -o -name 'steps_*_model.safetensors' \) -print 2>/dev/null)
  fallback="$(gb_to_bytes 80)"
  budget=$((state_max + weight_max))
  [[ "$budget" -lt "$fallback" ]] && budget="$fallback"
  echo "$budget"
}
threshold_bytes() { [[ "$1" == auto ]] && echo "" || gb_to_bytes "$1"; }
pid_matches_run() {
  local pid="${1:-}" cmd
  [[ -n "$pid" && -e "/proc/$pid" ]] || return 1
  if [[ -r "/proc/$pid/environ" ]] && tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -Fxq "STARVLA_JOB_DIR=$RUN_DIR"; then
    return 0
  fi
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmd" == *"$RUN_DIR"* || "$cmd" == *"$RUNNER"* ]]
}
alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null && pid_matches_run "$1"; }
pid_file() { [[ -f "$RUN_DIR/$1" ]] && tr -dc '0-9' < "$RUN_DIR/$1" || true; }
training_pids() { local p; for p in "$(pid_file main.pid)" "$(pid_file main.accelerate.pid)"; do alive "$p" && echo "$p"; done; }
is_running() { [[ -n "$(training_pids | head -1)" ]]; }
is_complete() { [[ -f "$FINAL_DIR/pytorch_model.pt" || -f "$FINAL_DIR/model.safetensors" ]]; }
detect_cuda_oom() { [[ -f "$MAIN_LOG" ]] && tail -c 1048576 "$MAIN_LOG" 2>/dev/null | grep -Eiq 'torch\.OutOfMemoryError|CUDA out of memory'; }

stop_training() {
  local reason="$1" pgid pids deadline pid
  pgid="$(pid_file main.accelerate.pgid)"
  pids="$(training_pids | xargs echo || true)"
  log "stopping training: $reason; pgid=${pgid:-none}; pids=${pids:-none}"
  event stop_requested "$reason"
  [[ "$DRY_RUN" == 1 ]] && return
  [[ -n "$pgid" && "$pgid" -gt 1 ]] && kill -TERM "-$pgid" 2>/dev/null || true
  for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + GRACE_SECONDS))
  while [[ "$SECONDS" -lt "$deadline" ]]; do is_running || { event stopped term; return; }; sleep 2; done
  log "force killing remaining training processes"
  [[ -n "$pgid" && "$pgid" -gt 1 ]] && kill -KILL "-$pgid" 2>/dev/null || true
  for pid in $(training_pids); do kill -KILL "$pid" 2>/dev/null || true; done
  event stopped kill
}
launch_training() {
  [[ -f "$RUNNER" ]] || { log "runner not found: $RUNNER"; exit 2; }
  if [[ "$MAX_RESTARTS" -gt 0 && "$RESTARTS" -ge "$MAX_RESTARTS" ]]; then log "max restarts reached: $MAX_RESTARTS"; exit 3; fi
  RESTARTS=$((RESTARTS + 1))
  log "launching training restart=$RESTARTS runner=$RUNNER latest_state=$(latest_state_step) latest_weight=$(latest_valid_weight_step)"
  event launch "restart=$RESTARTS"
  [[ "$DRY_RUN" == 1 ]] && return
  setsid bash "$RUNNER" >/dev/null 2>&1 &
}
status_line() {
  local free="$1" max_steps=0
  [[ -f "$JOB_JSON" ]] && max_steps="$(python - "$JOB_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("max_train_steps", 0))
PY
)"
  echo "watch=$WATCH_PATH free=$(human "$free") stop<$(human "$STOP_BYTES") resume>=$(human "$RESUME_BYTES") budget=$(human "$BUDGET_BYTES") latest_state=$(latest_state_step) latest_weight=$(latest_valid_weight_step) summary=$(latest_summary_step) max=$max_steps running=$(is_running && echo yes || echo no) complete=$(is_complete && echo yes || echo no)"
}

trap 'rm -f "$PID_FILE" 2>/dev/null || true' EXIT
echo "$$" > "$PID_FILE"
BUDGET_BYTES="$(estimate_budget_bytes)"
STOP_BYTES="$(threshold_bytes "$STOP_FREE_GB")"
RESUME_BYTES="$(threshold_bytes "$RESUME_FREE_GB")"
RESERVE_BYTES="$(gb_to_bytes 20)"
MIN_STOP_BYTES="$(gb_to_bytes 80)"
MARGIN_BYTES="$(gb_to_bytes 50)"
if [[ -z "$STOP_BYTES" ]]; then STOP_BYTES=$((BUDGET_BYTES * 125 / 100 + RESERVE_BYTES)); [[ "$STOP_BYTES" -lt "$MIN_STOP_BYTES" ]] && STOP_BYTES="$MIN_STOP_BYTES"; fi
if [[ -z "$RESUME_BYTES" ]]; then RESUME_BYTES=$((BUDGET_BYTES * 200 / 100 + RESERVE_BYTES)); min_resume=$((STOP_BYTES + MARGIN_BYTES)); [[ "$RESUME_BYTES" -lt "$min_resume" ]] && RESUME_BYTES="$min_resume"; fi
[[ "$RESUME_BYTES" -gt "$STOP_BYTES" ]] || { echo "[disk-guard] resume threshold must be greater than stop threshold" >&2; exit 2; }

log "disk guard started"
log "$(status_line "$(free_bytes)")"
find "$WEIGHT_DIR" -maxdepth 1 -type f -size -1073741824c -name 'steps_*' -print 2>/dev/null | while read -r p; do log "incomplete checkpoint ignored: $p size=$(file_size "$p")"; done
[[ "$STATUS_ONLY" == 1 ]] && exit 0

RESTARTS=0
while true; do
  free="$(free_bytes)"
  if is_complete; then log "training is complete; final_model exists at $FINAL_DIR"; event complete "$FINAL_DIR"; exit 0; fi
  if is_running; then
    if [[ "$free" -lt "$STOP_BYTES" ]]; then stop_training "free=$(human "$free") below stop=$(human "$STOP_BYTES")"; else event heartbeat "running free=$free"; fi
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ "$ABORT_ON_CUDA_OOM" == 1 ]] && detect_cuda_oom; then
    log "recent CUDA OOM detected in main.log; not relaunching blindly. Lower batch size/use more GPUs, or rerun with --no-abort-on-cuda-oom for disk-only retries."
    event abort_cuda_oom "$MAIN_LOG"
    exit 42
  fi
  if [[ "$free" -ge "$RESUME_BYTES" ]]; then launch_training; sleep 10; else log "waiting for free space: free=$(human "$free") resume>=$(human "$RESUME_BYTES") stop<$(human "$STOP_BYTES")"; event waiting_for_space "free=$free"; sleep "$POLL_SECONDS"; fi
done
