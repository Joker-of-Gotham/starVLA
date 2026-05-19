#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

yes=0
dry_run=0
signal_first="TERM"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)
      yes=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --sigkill)
      signal_first="KILL"
      shift
      ;;
    --help|-h)
      cat <<'EOF'
usage: bash interaction/bin/starvla-clean-gpu-orphans.sh [--dry-run] [--yes] [--sigkill]

Find CUDA compute processes that look like StarVLA training/eval workers and
terminate them.  This is intended for failed accelerate/deepspeed launches that
leave python workers holding H200 memory with 0% GPU util.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python - "$yes" "$dry_run" "$signal_first" <<'PY'
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

YES = sys.argv[1] == "1"
DRY_RUN = sys.argv[2] == "1"
SIGNAL_FIRST = signal.SIGKILL if sys.argv[3] == "KILL" else signal.SIGTERM
REPO = Path.cwd().resolve()
ENV_MARK = "/env/starvla-py310/bin/python"


def nvidia_smi() -> str | None:
    for candidate in (
        "nvidia-smi",
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/opt/nvidia/bin/nvidia-smi",
    ):
        try:
            proc = subprocess.run([candidate, "-L"], text=True, capture_output=True, timeout=5)
        except Exception:
            continue
        if proc.returncode == 0:
            return candidate
    return None


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def environ(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return b""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


binary = nvidia_smi()
if not binary:
    print("[starvla-clean] nvidia-smi not found", file=sys.stderr)
    raise SystemExit(127)

query = [
    binary,
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
]
proc = subprocess.run(query, text=True, capture_output=True, timeout=10)
if proc.returncode != 0:
    print(proc.stdout + proc.stderr, file=sys.stderr)
    raise SystemExit(proc.returncode)

by_pid: dict[int, dict] = {}
for line in proc.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        continue
    try:
        pid = int(float(parts[1]))
        mem = int(float(parts[3]))
    except ValueError:
        continue
    item = by_pid.setdefault(pid, {"pid": pid, "process_name": parts[2], "memory": 0, "gpus": 0})
    item["memory"] += mem
    item["gpus"] += 1

matches = []
for pid, item in sorted(by_pid.items()):
    if pid in {os.getpid(), os.getppid()}:
        continue
    cmd = cmdline(pid)
    env = environ(pid)
    process_name = str(item["process_name"])
    looks_starvla = (
        ENV_MARK in process_name
        or ENV_MARK in cmd
        or "starVLA/training/" in cmd
        or (str(REPO) in cmd and "accelerate" in cmd)
        or b"STARVLA_JOB_DIR=" in env
    )
    if looks_starvla:
        item["cmd"] = cmd or process_name
        matches.append(item)

print(f"[starvla-clean] matched StarVLA CUDA worker pids={len(matches)}")
for item in matches:
    short = item["cmd"][:220]
    print(f"pid={item['pid']} gpus={item['gpus']} mem={item['memory']}MiB cmd={short}")

if not matches:
    raise SystemExit(0)

if DRY_RUN:
    print("[starvla-clean] dry-run only; no process was killed")
    raise SystemExit(0)

if not YES:
    answer = input("Kill these StarVLA GPU processes? Type yes to continue: ").strip().lower()
    if answer not in {"yes", "y"}:
        print("[starvla-clean] cancelled")
        raise SystemExit(130)

pids = [int(item["pid"]) for item in matches]
for sig, delay in [(SIGNAL_FIRST, 3.0), (signal.SIGKILL, 0.0)]:
    for pid in sorted(pids, reverse=True):
        if alive(pid):
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    if delay:
        time.sleep(delay)

remaining = [pid for pid in pids if alive(pid)]
print(f"[starvla-clean] remaining pids={remaining}")
if remaining:
    raise SystemExit(1)
PY
