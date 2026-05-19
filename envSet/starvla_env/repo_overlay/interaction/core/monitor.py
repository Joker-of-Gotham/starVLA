from __future__ import annotations

import ast
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

from .diagnostics import compact_value, diagnose_log, important_metrics, read_last_jsonl, safe_float
from .paths import candidate_runs_roots
from .tmux import session_exists
from .ui import clear_and_print, human_age, human_bytes, severity_style

STEP_RE = re.compile(r"Step\s+(\d+),\s+(?:Loss|Metrics):\s+(\{.*\})\)?")
TQDM_RE = re.compile(r"(\d+(?:\.\d+)?)%\|.*?\|\s*(\d+)/(\d+)")
ERROR_LINE_RE = re.compile(r"(error|exception|traceback|failed|fatal|out of memory|nan|inf|warning)", re.I)


def iter_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for runs_root in candidate_runs_roots():
        for meta_path in sorted(runs_root.glob("*/job.json"), reverse=True):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta["meta_path"] = str(meta_path)
            meta["session_alive"] = session_exists(meta.get("tmux_session", meta.get("job_name", "")))
            meta["latest"] = parse_latest_metrics(Path(meta.get("log_file", "")), meta.get("max_train_steps"))
            events = read_events(meta_path.parent, lines=40)
            meta["latest_event"] = events[-1] if events else {}
            meta["diagnosis"] = diagnose_log(tail(Path(meta.get("log_file", "")), lines=400), diagnostic_event(events))
            jobs.append(meta)
    return jobs


def find_job(name_or_dir: str) -> dict[str, Any] | None:
    direct = Path(name_or_dir)
    candidates: list[Path] = []
    if direct.exists():
        candidates.append(direct / "job.json" if direct.is_dir() else direct)
    else:
        for runs_root in candidate_runs_roots():
            candidates.extend(runs_root.glob(f"*_{name_or_dir}/job.json"))
            candidates.extend(path for path in runs_root.glob("*/job.json") if name_or_dir in path.parent.name)

    unique_candidates = sorted(set(candidates), key=lambda path: path.parent.name, reverse=True)
    for meta_path in unique_candidates:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["meta_path"] = str(meta_path)
            return meta
    return None


def read_events(job_dir: Path, lines: int = 40) -> list[dict[str, Any]]:
    return read_last_jsonl(job_dir / "events.jsonl", lines=lines)


def latest_event(job_dir: Path) -> dict[str, Any]:
    events = read_events(job_dir, lines=20)
    return events[-1] if events else {}


def diagnostic_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") == "failed":
            return event
    return events[-1] if events else {}


def tail(path: Path, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return list(deque(f, maxlen=max(lines, 1)))
    except Exception:
        return []


def parse_latest_metrics(log_file: Path, max_steps: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"step": None, "max_steps": max_steps, "epoch": None, "metrics": {}, "progress": None}
    if not log_file.exists():
        return result
    for line in tail(log_file, lines=800):
        step_match = STEP_RE.search(line)
        if step_match:
            result["step"] = int(step_match.group(1))
            try:
                metrics = ast.literal_eval(step_match.group(2))
                if isinstance(metrics, dict):
                    result["metrics"] = metrics
                    result["epoch"] = metrics.get("epoch")
            except Exception:
                pass
        tqdm_match = TQDM_RE.search(line)
        if tqdm_match:
            result["progress"] = {
                "percent": float(tqdm_match.group(1)),
                "current": int(tqdm_match.group(2)),
                "total": int(tqdm_match.group(3)),
            }
    if result["step"] is not None and max_steps:
        result["progress"] = {"percent": round(result["step"] * 100 / max_steps, 2), "current": result["step"], "total": max_steps}
    return result


def collect_job_snapshot(job: dict[str, Any], tail_lines: int = 18, diagnostic_lines: int = 800) -> dict[str, Any]:
    job_dir = Path(job.get("job_dir") or Path(job.get("meta_path", ".")).parent)
    session = job.get("tmux_session", job.get("job_name", ""))
    log_files = _job_log_files(job)
    primary_log = Path(job.get("log_file") or (log_files[0][1] if log_files else ""))
    latest = parse_latest_metrics(primary_log, job.get("max_train_steps"))
    events = read_events(job_dir, lines=80)
    event = events[-1] if events else {}
    diagnostic_log: list[str] = []
    for _, path in log_files:
        diagnostic_log.extend(tail(Path(path), lines=max(80, diagnostic_lines // max(len(log_files), 1))))
    diagnosis = diagnose_log(diagnostic_log, diagnostic_event(events))
    log_tail = [(name, str(path), tail(Path(path), lines=tail_lines)) for name, path in log_files]
    checkpoint = checkpoint_report(job)
    primary_stat = _stat(primary_log)
    return {
        "job": job,
        "job_dir": str(job_dir),
        "session": session,
        "session_alive": session_exists(session),
        "latest": latest,
        "events": events,
        "latest_event": event,
        "diagnosis": diagnosis,
        "checkpoint": checkpoint,
        "log_files": [(name, str(path)) for name, path in log_files],
        "log_tail": log_tail,
        "primary_log_size": primary_stat.get("size"),
        "primary_log_mtime": primary_stat.get("mtime"),
        "gpu_snapshot": parse_gpu_snapshot(event.get("gpu_snapshot") or []),
    }


def checkpoint_report(job: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "output_dir": job.get("output_dir"),
        "checkpoint_dir": job.get("checkpoint_dir"),
        "full_state_dir": job.get("full_state_dir"),
        "summary_jsonl": job.get("summary_jsonl"),
        "weight_count": 0,
        "state_count": 0,
        "topk": [],
        "latest_summary": {},
        "latest_state": None,
    }
    checkpoint_dir_value = job.get("checkpoint_dir")
    checkpoint_dir = Path(str(checkpoint_dir_value)) if checkpoint_dir_value else None
    if checkpoint_dir and checkpoint_dir.exists():
        report["weight_count"] = len([path for path in checkpoint_dir.iterdir() if path.is_file()])
    full_state_value = job.get("full_state_dir")
    full_state_dir = Path(str(full_state_value)) if full_state_value else None
    if full_state_dir and full_state_dir.exists():
        report["state_count"] = len([path for path in full_state_dir.iterdir() if path.is_dir() and path.name.startswith("steps_")])
        latest = full_state_dir / "latest"
        if latest.exists() or latest.is_symlink():
            report["latest_state"] = str(latest)
    output_dir_value = job.get("output_dir")
    topk_path = Path(str(output_dir_value)) / "topk_checkpoints.json" if output_dir_value else None
    if topk_path and topk_path.exists():
        try:
            data = json.loads(topk_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                report["topk"] = data.get("checkpoints") or data.get("topk") or []
            elif isinstance(data, list):
                report["topk"] = data
        except Exception as exc:
            report["topk_error"] = repr(exc)
    summary_value = job.get("summary_jsonl")
    if summary_value:
        for item in read_last_jsonl(Path(str(summary_value)), lines=1):
            report["latest_summary"] = item
    return report


def parse_gpu_snapshot(lines: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in lines:
        parts = [part.strip() for part in str(line).split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "id": parts[0],
                "uuid": parts[1],
                "name": parts[2],
                "used": parts[3],
                "total": parts[4],
                "util": parts[5],
                "power": parts[6] if len(parts) > 6 else "-",
                "power_limit": parts[7] if len(parts) > 7 else "-",
                "pstate": parts[8] if len(parts) > 8 else "-",
                "temp": parts[9] if len(parts) > 9 else "-",
                "sm_clock": parts[10] if len(parts) > 10 else "-",
                "mem_clock": parts[11] if len(parts) > 11 else "-",
            }
        )
    return rows


def watch(job: dict[str, Any], interval: float = 2.0, tail_lines: int = 18) -> None:
    try:
        from rich.console import Console
        from rich.live import Live
    except Exception:
        _plain_watch(job, interval, tail_lines)
        return

    console = Console(highlight=False)
    if not console.is_terminal:
        _plain_watch(job, interval, tail_lines)
        return

    def render() -> Any:
        return render_snapshot(collect_job_snapshot(job, tail_lines=tail_lines))

    refresh = max(1, min(8, int(round(1 / max(interval, 0.2)))))
    try:
        with Live(render(), console=console, refresh_per_second=refresh, transient=False, screen=False) as live:
            while True:
                time.sleep(interval)
                live.update(render())
    except KeyboardInterrupt:
        console.print("[yellow]monitor stopped[/yellow]")


def render_snapshot(snapshot: dict[str, Any]) -> Any:
    from rich import box
    from rich.columns import Columns
    from rich.console import Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text

    job = snapshot["job"]
    latest = snapshot["latest"]
    event = snapshot["latest_event"]
    diagnosis = snapshot["diagnosis"]
    severity = diagnosis.get("severity", "ok")
    title = f"StarVLA {job.get('kind', 'job')} monitor - {job.get('job_name', job.get('run_id', 'unknown'))}"

    overview = Table(show_header=False, box=None, expand=True, padding=(0, 1))
    overview.add_column("item", justify="right", style="cyan", no_wrap=True)
    overview.add_column("value", ratio=1, overflow="fold")
    state = "alive" if snapshot["session_alive"] else "stopped"
    state_style = "green" if snapshot["session_alive"] else "red"
    overview.add_row("session:", f"[{state_style}]{snapshot['session']} ({state})[/{state_style}]")
    overview.add_row("stage:", f"[{severity_style(severity)}]{diagnosis.get('stage', '-') or '-'}[/]")
    overview.add_row("event:", _event_text(event))
    overview.add_row("step:", _step_text(latest))
    overview.add_row("log:", f"{snapshot['log_files'][0][1] if snapshot['log_files'] else '-'}")
    overview.add_row("log age:", f"{human_age(snapshot.get('primary_log_mtime'))} / {human_bytes(snapshot.get('primary_log_size'))}")
    overview.add_row("job dir:", snapshot["job_dir"])

    progress_renderable: Any
    progress = latest.get("progress") or {}
    if progress.get("total"):
        progress_bar = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            expand=True,
        )
        progress_bar.add_task(
            f"{progress.get('current', 0)}/{progress.get('total', 0)}",
            total=float(progress.get("total") or 1),
            completed=float(progress.get("current") or 0),
        )
        progress_renderable = progress_bar
    else:
        progress_renderable = Text("No step progress parsed yet.", style="yellow")

    metrics_table = Table(title="Metrics", box=box.SIMPLE, expand=True)
    metrics_table.add_column("name", style="cyan", no_wrap=True)
    metrics_table.add_column("value", overflow="fold")
    metric_rows = important_metrics(latest.get("metrics") or {})
    if metric_rows:
        for key, value in metric_rows:
            value_style = _metric_style(key, value)
            metrics_table.add_row(key, f"[{value_style}]{compact_value(value)}[/{value_style}]")
    else:
        metrics_table.add_row("-", "[yellow]waiting for first metric line[/yellow]")

    diag_table = Table(title="Diagnostics", box=box.SIMPLE, expand=True)
    diag_table.add_column("code", style="cyan", no_wrap=True)
    diag_table.add_column("stage", no_wrap=True)
    diag_table.add_column("evidence", overflow="fold")
    issues = diagnosis.get("issues") or []
    if issues:
        for issue in issues[:4]:
            style = severity_style(issue.get("severity"))
            diag_table.add_row(
                f"[{style}]{issue.get('code')}[/]",
                str(issue.get("stage", "-")),
                escape(str(issue.get("evidence", "-"))),
            )
    else:
        diag_table.add_row("[green]OK[/green]", str(diagnosis.get("stage", "-")), "No known failure pattern in the recent log window.")

    checkpoint_table = Table(title="Artifacts", box=box.SIMPLE, expand=True)
    checkpoint_table.add_column("item", style="cyan", no_wrap=True)
    checkpoint_table.add_column("value", overflow="fold")
    checkpoint = snapshot["checkpoint"]
    checkpoint_table.add_row("weights", str(checkpoint.get("weight_count", 0)))
    checkpoint_table.add_row("states", str(checkpoint.get("state_count", 0)))
    checkpoint_table.add_row("top-k", _topk_text(checkpoint.get("topk") or []))
    checkpoint_table.add_row("latest state", str(checkpoint.get("latest_state") or "-"))
    summary = checkpoint.get("latest_summary") or {}
    if summary:
        checkpoint_table.add_row("summary", f"step={summary.get('steps', summary.get('step', '-'))} loss={compact_value(summary.get('loss', '-'))}")

    gpu_table = Table(title="GPU Snapshot", box=box.SIMPLE, expand=True)
    gpu_table.add_column("id", style="cyan", no_wrap=True)
    gpu_table.add_column("name", overflow="fold")
    gpu_table.add_column("mem")
    gpu_table.add_column("util")
    gpu_table.add_column("power")
    gpu_table.add_column("clk")
    if snapshot.get("gpu_snapshot"):
        for row in snapshot["gpu_snapshot"]:
            gpu_table.add_row(
                row["id"],
                row["name"],
                f"{row['used']}/{row['total']} MB",
                f"{row['util']}%",
                f"{row.get('power', '-')}/{row.get('power_limit', '-')}W {row.get('pstate', '-')}",
                f"sm={row.get('sm_clock', '-')} mem={row.get('mem_clock', '-')}",
            )
    else:
        gpu_table.add_row("-", "no nvidia-smi snapshot", "-", "-", "-", "-")

    log_text = _render_log_tail(snapshot["log_tail"])
    traceback_lines = diagnosis.get("traceback") or []
    panels: list[Any] = [
        Panel(overview, title=title, border_style=severity_style(severity)),
        Panel(progress_renderable, title="Progress", border_style="cyan"),
        Columns([metrics_table, checkpoint_table], equal=True, expand=True),
        Columns([diag_table, gpu_table], equal=True, expand=True),
        Panel(log_text, title="Recent Log Tail", border_style="blue"),
    ]
    if traceback_lines:
        panels.append(Panel(Text("\n".join(traceback_lines[-18:]), style="white"), title="Last Traceback", border_style="red"))
    if issues and issues[0].get("hints"):
        hints = "\n".join(f"- {hint}" for hint in issues[0]["hints"][:4])
        panels.append(Panel(hints, title="Immediate Fix Hints", border_style="yellow"))
    utilization_hints = _utilization_hints(snapshot)
    if utilization_hints:
        panels.append(Panel("\n".join(f"- {hint}" for hint in utilization_hints), title="Utilization Hints", border_style="yellow"))
    return Group(*panels)


def snapshot_to_text(snapshot: dict[str, Any]) -> str:
    job = snapshot["job"]
    latest = snapshot["latest"]
    diagnosis = snapshot["diagnosis"]
    checkpoint = snapshot["checkpoint"]
    event = snapshot["latest_event"]
    lines = [
        f"StarVLA {job.get('kind', 'job')} monitor - {job.get('job_name', job.get('run_id', 'unknown'))}",
        f"session: {snapshot['session']} ({'alive' if snapshot['session_alive'] else 'stopped'})",
        f"stage: {diagnosis.get('stage', '-')}, severity: {diagnosis.get('severity', '-')}, event: {_plain_event(event)}",
        f"step: {_plain_step(latest)}",
        f"log: {snapshot['log_files'][0][1] if snapshot['log_files'] else '-'} ({human_age(snapshot.get('primary_log_mtime'))}, {human_bytes(snapshot.get('primary_log_size'))})",
        f"artifacts: weights={checkpoint.get('weight_count', 0)}, states={checkpoint.get('state_count', 0)}, topk={len(checkpoint.get('topk') or [])}",
        "",
        "metrics:",
    ]
    metric_rows = important_metrics(latest.get("metrics") or {})
    lines.extend(f"  {key}: {compact_value(value)}" for key, value in metric_rows[:16])
    if not metric_rows:
        lines.append("  waiting for first metric line")
    lines += ["", "diagnostics:"]
    issues = diagnosis.get("issues") or []
    if issues:
        for issue in issues[:5]:
            lines.append(f"  {issue.get('code')} [{issue.get('stage')}]: {issue.get('evidence')}")
            for hint in issue.get("hints", [])[:2]:
                lines.append(f"    fix: {hint}")
    else:
        lines.append("  OK: no known failure pattern in the recent log window")
    lines += ["", "recent log tail:"]
    for name, _, log_lines in snapshot["log_tail"]:
        lines.append(f"  --- {name} ---")
        lines.extend("  " + line.rstrip("\n") for line in log_lines[-12:])
    return "\n".join(lines)


def _plain_watch(job: dict[str, Any], interval: float, tail_lines: int) -> None:
    try:
        while True:
            clear_and_print(snapshot_to_text(collect_job_snapshot(job, tail_lines=tail_lines)))
            time.sleep(interval)
    except KeyboardInterrupt:
        clear_and_print("monitor stopped")


def _utilization_hints(snapshot: dict[str, Any]) -> list[str]:
    rows = snapshot.get("gpu_snapshot") or []
    if not rows or not snapshot.get("session_alive"):
        return []
    utils = [_num(row.get("util")) for row in rows]
    power = [_num(row.get("power")) for row in rows]
    power_limit = [_num(row.get("power_limit")) for row in rows]
    used = [_num(row.get("used")) for row in rows]
    total = [_num(row.get("total")) for row in rows]
    avg_util = _avg([value for value in utils if value is not None])
    avg_power_pct = _avg(
        [
            value * 100 / limit
            for value, limit in zip(power, power_limit)
            if value is not None and limit not in {None, 0}
        ]
    )
    avg_mem_pct = _avg(
        [
            value * 100 / limit
            for value, limit in zip(used, total)
            if value is not None and limit not in {None, 0}
        ]
    )
    hints: list[str] = []
    if avg_util is not None and avg_util < 55:
        hints.append(f"Average GPU util is {avg_util:.0f}%; check dataloader/I/O, video decode backend, CPU workers, or increase per-device batch size.")
    if avg_power_pct is not None and avg_power_pct < 45 and (avg_util is None or avg_util < 70):
        hints.append(f"Average power draw is {avg_power_pct:.0f}% of limit; GPUs are probably waiting rather than compute-bound.")
    if avg_mem_pct is not None and avg_mem_pct < 55:
        current_vla_batch = _num(snapshot.get("job", {}).get("vla_batch_size"))
        current_vlm_batch = _num(snapshot.get("job", {}).get("vlm_batch_size"))
        next_vla = int(current_vla_batch * 2) if current_vla_batch else None
        next_vlm = int(current_vlm_batch * 2) if current_vlm_batch else None
        if next_vla:
            hints.append(f"Average memory use is {avg_mem_pct:.0f}%; next H200 tuning run can try --vla-batch-size {next_vla}" + (f" --vlm-batch-size {next_vlm}" if next_vlm else "") + ".")
        else:
            hints.append(f"Average memory use is {avg_mem_pct:.0f}%; H200 has room to raise VLA/VLM batch size if loss behavior remains stable.")
    job = snapshot.get("job", {})
    if job.get("throughput_profile") not in {"h200_saturated", "auto"} and len(rows) >= 8:
        hints.append("8 GPUs are active without h200_saturated throughput profile; relaunch with --throughput-profile h200_saturated after saving/resuming the run.")
    return hints[:3]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().split()[0]
    try:
        return float(text)
    except ValueError:
        return None


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _job_log_files(job: dict[str, Any]) -> list[tuple[str, Path]]:
    candidates = [
        ("all", job.get("combined_log")),
        ("main", job.get("log_file")),
        ("server", job.get("server_log")),
        ("client", job.get("client_log")),
    ]
    window_logs = job.get("window_logs")
    if isinstance(window_logs, dict):
        candidates.extend((str(name), value) for name, value in sorted(window_logs.items()))
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for name, value in candidates:
        if not value:
            continue
        path = Path(str(value))
        if path in seen:
            continue
        seen.add(path)
        out.append((name, path))
    return out


def _stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def _event_text(event: dict[str, Any]) -> str:
    if not event:
        return "[yellow]no event[/yellow]"
    status = str(event.get("status", "0"))
    style = "green" if status == "0" and event.get("event") != "failed" else "red"
    detail = event.get("detail")
    text = f"{event.get('event', '-')} status={status} time={event.get('time', '-')}"
    if detail:
        text += f" detail={detail}"
    return f"[{style}]{text}[/{style}]"


def _plain_event(event: dict[str, Any]) -> str:
    if not event:
        return "no event"
    detail = f" detail={event.get('detail')}" if event.get("detail") else ""
    return f"{event.get('event', '-')} status={event.get('status', '-')} time={event.get('time', '-')}{detail}"


def _step_text(latest: dict[str, Any]) -> str:
    progress = latest.get("progress") or {}
    if progress:
        return f"{progress.get('current', '-')}/{progress.get('total', '-')} ({progress.get('percent', '-')}%), epoch={compact_value(latest.get('epoch'))}"
    return f"{latest.get('step', '-')}, epoch={compact_value(latest.get('epoch'))}"


def _plain_step(latest: dict[str, Any]) -> str:
    return _step_text(latest).replace("[", "").replace("]", "")


def _metric_style(key: str, value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "white"
    if not (-1e20 < number < 1e20):
        return "red"
    key_lower = key.lower()
    if "loss" in key_lower or "mse" in key_lower:
        return "green" if number >= 0 else "yellow"
    if "lr" in key_lower:
        return "magenta"
    if "timing" in key_lower:
        return "blue"
    return "white"


def _topk_text(items: list[Any]) -> str:
    if not items:
        return "-"
    parts: list[str] = []
    for index, item in enumerate(items[:3], start=1):
        if isinstance(item, dict):
            step = item.get("steps", item.get("step", "-"))
            loss = compact_value(item.get("loss", "-"))
            parts.append(f"#{index}: step={step} loss={loss}")
        else:
            parts.append(f"#{index}: {compact_value(item)}")
    return "; ".join(parts)


def _render_log_tail(log_tail: list[tuple[str, str, list[str]]]) -> Any:
    from rich.text import Text

    text = Text()
    for name, path, lines in log_tail:
        text.append(f"--- {name}: {path} ---\n", style="bold cyan")
        if not lines:
            text.append("(empty or missing)\n", style="yellow")
            continue
        for line in lines:
            style = "red" if ERROR_LINE_RE.search(line) else "white"
            if "Step " in line and ("Loss:" in line or "Metrics:" in line):
                style = "green"
            text.append(line.rstrip("\n") + "\n", style=style)
    return text
