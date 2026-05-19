#!/usr/bin/env bash
set -Eeuo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

job="${1:-}"
if [[ -z "$job" ]]; then
  echo "usage: bash interaction/bin/starvla-force-stop.sh <job-name-or-run-dir>" >&2
  exit 2
fi

python - "$job" <<'PY'
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from interaction.core.monitor import find_job
from interaction.core.tmux import kill_session

arg = sys.argv[1]
job = find_job(arg)
if job:
    session = job.get("tmux_session") or job.get("job_name") or arg
    job_dir = Path(job.get("job_dir") or Path(job.get("meta_path", ".")).parent)
    ckpt = str(job.get("ckpt") or "")
    ports = {str(x) for x in (job.get("ports") or []) if x is not None}
    if job.get("port") is not None:
        ports.add(str(job.get("port")))
else:
    session = Path(arg).name
    job_dir = Path(arg)
    ckpt = ""
    ports = set()

print(f"[starvla-force-stop] session={session}")
print(f"[starvla-force-stop] job_dir={job_dir}")
kill_session(session, job_dir=job_dir if job_dir.exists() else None, grace_seconds=1.5)

def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def job_env_pids() -> set[int]:
    if not job_dir.exists():
        return set()
    target = str(job_dir.resolve())
    target_env = f"STARVLA_JOB_DIR={target}".encode()
    target_name = job_dir.name.encode()
    out: set[int] = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid <= 1 or pid == os.getpid():
            continue
        try:
            env = (proc / "environ").read_bytes()
        except OSError:
            continue
        if target_env in env or (b"STARVLA_JOB_DIR=" in env and target_name in env):
            out.add(pid)
    return out

def nvidia_pids() -> set[int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return set()
    out = set()
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            out.add(int(text))
    return out

def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""

def eval_cmdline_pids() -> set[int]:
    out: set[int] = set()
    job_text = str(job_dir.resolve()) if job_dir.exists() else ""
    job_name = job_dir.name
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid <= 1 or pid == os.getpid():
            continue
        cmd = cmdline(pid)
        if not cmd:
            continue
        is_eval = "deployment/model_server/server_policy.py" in cmd or "examples/calvin/eval_files/eval_calvin.py" in cmd
        if not is_eval:
            continue
        if job_text and job_text in cmd:
            out.add(pid)
            continue
        if job_name and job_name in cmd:
            out.add(pid)
            continue
        if ckpt and ckpt in cmd and (not ports or any(f"--port {port}" in cmd or f"--args.port {port}" in cmd for port in ports)):
            out.add(pid)
            continue
        if ports and any(f"--port {port}" in cmd or f"--args.port {port}" in cmd for port in ports):
            out.add(pid)
    return out

pids = {pid for pid in (job_env_pids() | eval_cmdline_pids()) if alive(pid)}
print(f"[starvla-force-stop] matched alive eval pids={len(pids)}")
for sig, delay in [(signal.SIGTERM, 1.5), (signal.SIGKILL, 0.0)]:
    for pid in sorted(pids, reverse=True):
        if alive(pid):
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    if delay:
        time.sleep(delay)

remaining = {pid for pid in (job_env_pids() | eval_cmdline_pids()) if alive(pid)}
gpu_remaining = nvidia_pids() & remaining
print(f"[starvla-force-stop] remaining matched pids={len(remaining)}")
print(f"[starvla-force-stop] remaining matched gpu pids={sorted(gpu_remaining)[:40]}")
if gpu_remaining:
    raise SystemExit(1)
PY
