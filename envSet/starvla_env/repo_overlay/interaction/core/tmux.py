from __future__ import annotations

import os
import signal
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from .paths import REPO_ROOT


@lru_cache(maxsize=1)
def tmux_binary() -> str | None:
    candidates: list[str] = []
    env_value = os.environ.get("STARVLA_TMUX")
    if env_value:
        candidates.append(env_value)
    found = shutil.which("tmux")
    if found:
        candidates.append(found)
    candidates.extend(["/usr/bin/tmux", "/bin/tmux"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _tmux_works(candidate):
            return candidate
    return None


def _tmux_works(candidate: str) -> bool:
    try:
        proc = subprocess.run([candidate, "-V"], capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "tmux" in (proc.stdout + proc.stderr).lower()


def tmux_available() -> bool:
    return tmux_binary() is not None


def _tmux_cmd() -> str:
    binary = tmux_binary()
    if not binary:
        raise RuntimeError("tmux is not installed or no working tmux binary is available. Set STARVLA_TMUX=/path/to/tmux if needed.")
    return binary


def session_exists(session: str) -> bool:
    binary = tmux_binary()
    if not binary:
        return False
    return subprocess.run([binary, "has-session", "-t", session], capture_output=True).returncode == 0


def new_session(session: str, runner: Path, cwd: Path = REPO_ROOT, window_name: str = "main") -> None:
    binary = _tmux_cmd()
    if session_exists(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    subprocess.run(
        [binary, "new-session", "-d", "-s", session, "-n", window_name, "-c", str(cwd), "bash", str(runner)],
        check=True,
    )


def new_window(session: str, name: str, runner: Path, cwd: Path = REPO_ROOT) -> None:
    binary = _tmux_cmd()
    if not session_exists(session):
        raise RuntimeError(f"tmux session does not exist: {session}")
    subprocess.run([binary, "new-window", "-t", session, "-n", name, "-c", str(cwd), "bash", str(runner)], check=True)


def attach(session: str) -> None:
    subprocess.run([_tmux_cmd(), "attach-session", "-t", session], check=False)


def kill_session(session: str, job_dir: Path | str | None = None, grace_seconds: float = 5.0) -> None:
    binary = tmux_binary()
    pids: set[int] = set()
    if binary and session_exists(session):
        pids.update(_tmux_pane_pids(binary, session))
    if job_dir is not None:
        job_path = Path(job_dir)
        pids.update(_job_pids(job_path))
        pids.update(_job_env_pids(job_path))
        process_groups = _job_pgids(job_path)
    else:
        process_groups = set()

    pids = {pid for pid in pids if _pid_alive(pid)}
    process_tree: set[int] = set()
    for pid in pids:
        process_tree.update(_process_tree(pid))
    discovered_groups = {_process_group(pid) for pid in process_tree | pids}
    process_groups.update(pgid for pgid in discovered_groups if pgid)
    process_groups = {pgid for pgid in process_groups if pgid > 1 and pgid != os.getpgrp()}
    if process_tree:
        _signal_pids(process_tree, signal.SIGTERM)
    if process_groups:
        _signal_pgroups(process_groups, signal.SIGTERM)

    if binary and session_exists(session):
        subprocess.run([binary, "kill-session", "-t", session], check=False)

    if process_tree:
        deadline = time.time() + max(float(grace_seconds), 0.0)
        while time.time() < deadline and any(_pid_alive(pid) for pid in process_tree):
            time.sleep(0.2)
        survivors = {pid for pid in process_tree if _pid_alive(pid)}
        if survivors:
            _signal_pids(survivors, signal.SIGKILL)
    if process_groups:
        _signal_pgroups(process_groups, signal.SIGKILL)


def _tmux_pane_pids(binary: str, session: str) -> set[int]:
    proc = subprocess.run(
        [binary, "list-panes", "-t", session, "-a", "-F", "#{pane_pid}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()
    return {int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()}


def _job_pids(job_dir: Path) -> set[int]:
    if not job_dir.exists():
        return set()
    pids: set[int] = set()
    for pid_file in job_dir.rglob("*.pid"):
        try:
            text = pid_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit():
            pids.add(int(text))
    return pids


def _job_pgids(job_dir: Path) -> set[int]:
    if not job_dir.exists():
        return set()
    pgids: set[int] = set()
    for pgid_file in job_dir.rglob("*.pgid"):
        try:
            text = pgid_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit():
            pgids.add(int(text))
    return pgids


def _job_env_pids(job_dir: Path) -> set[int]:
    """Find orphaned descendants that inherited STARVLA_JOB_DIR."""
    if not job_dir.exists():
        return set()
    try:
        target = str(job_dir.resolve())
    except OSError:
        target = str(job_dir)
    target_env = f"STARVLA_JOB_DIR={target}".encode()
    target_basename = job_dir.name.encode()
    pids: set[int] = set()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return pids
    current_pid = os.getpid()
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        if pid <= 1 or pid == current_pid:
            continue
        try:
            environ = (path / "environ").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if target_env in environ or (b"STARVLA_JOB_DIR=" in environ and target_basename in environ):
            pids.add(pid)
    return pids


def _child_pids(pid: int) -> set[int]:
    proc = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True)
    if proc.returncode != 0:
        return set()
    return {int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()}


def _process_group(pid: int) -> int | None:
    proc = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return int(text) if text.isdigit() else None


def _process_tree(pid: int) -> set[int]:
    if pid <= 1:
        return set()
    found: set[int] = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in found or current <= 1:
            continue
        found.add(current)
        stack.extend(_child_pids(current))
    return found


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_pids(pids: set[int], sig: signal.Signals) -> None:
    for pid in sorted(pids, reverse=True):
        if pid <= 1:
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def _signal_pgroups(pgids: set[int], sig: signal.Signals) -> None:
    for pgid in sorted(pgids, reverse=True):
        if pgid <= 1 or pgid == os.getpgrp():
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def list_sessions() -> list[str]:
    binary = tmux_binary()
    if not binary:
        return []
    proc = subprocess.run([binary, "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
