from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
METRIC_NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class FailurePattern:
    code: str
    severity: str
    stage: str
    pattern: re.Pattern[str]
    headline: str
    hints: tuple[str, ...]


FAILURE_PATTERNS: tuple[FailurePattern, ...] = (
    FailurePattern(
        "cuda_oom",
        "fatal",
        "gpu-memory",
        re.compile(r"(CUDA out of memory|CUDACachingAllocator|DefaultCPUAllocator.*allocate|OutOfMemoryError)", re.I),
        "GPU/CPU memory is exhausted.",
        (
            "Reduce per-device batch size or sequence/video length.",
            "Use fewer concurrent jobs on the selected GPUs, or select GPUs with more free memory.",
            "If fragmentation is suspected, keep PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
        ),
    ),
    FailurePattern(
        "nccl_or_distributed",
        "fatal",
        "distributed",
        re.compile(r"(NCCL|ProcessGroupNCCL|DistributedDataParallel|torch\.distributed|Rendezvous|Connection reset by peer|Socket Timeout)", re.I),
        "Distributed/NCCL communication failed.",
        (
            "Check that every selected GPU is visible to the same tmux process.",
            "Verify num_processes equals the selected CUDA_VISIBLE_DEVICES count.",
            "Look for a rank that failed earlier; the NCCL error is often a secondary symptom.",
        ),
    ),
    FailurePattern(
        "missing_path",
        "fatal",
        "filesystem",
        re.compile(r"(No such file or directory|FileNotFoundError|cannot find|not found at|does not exist)", re.I),
        "A required file, checkpoint, model, or dataset path is missing.",
        (
            "Check the path printed in the evidence line.",
            "Run the interaction check command to validate configured datasets and pretrained models.",
            "For resume jobs, verify the chosen accelerate_state or checkpoint path exists.",
        ),
    ),
    FailurePattern(
        "permission",
        "fatal",
        "filesystem",
        re.compile(r"(PermissionError|permission denied|Read-only file system|Operation not permitted)", re.I),
        "The process cannot write or read a required path.",
        (
            "Move --run-root-dir to a writable directory or fix filesystem permissions.",
            "Check checkpoint_root and interaction runs_root writability.",
        ),
    ),
    FailurePattern(
        "disk_full",
        "fatal",
        "filesystem",
        re.compile(r"(No space left on device|Disk quota exceeded|quota exceeded)", re.I),
        "The output filesystem is full or over quota.",
        (
            "Free space under the checkpoint/run root.",
            "Reduce --keep-top-k or disable full state snapshots only if resume state is not needed.",
        ),
    ),
    FailurePattern(
        "import_error",
        "fatal",
        "environment",
        re.compile(r"(ModuleNotFoundError|ImportError|undefined symbol|cannot import name)", re.I),
        "Python environment is missing a package or has an ABI mismatch.",
        (
            "Confirm the tmux runner sourced bar/activate_starvla_no_proxy.sh.",
            "Run the interaction check command and inspect the Python imports table.",
        ),
    ),
    FailurePattern(
        "cuda_kernel",
        "fatal",
        "cuda-runtime",
        re.compile(r"(CUDA error|device-side assert|illegal memory access|CUBLAS_STATUS|cusolver|cuDNN error)", re.I),
        "CUDA runtime/kernel failure.",
        (
            "Inspect the first traceback above this line; later CUDA errors may be delayed symptoms.",
            "Rerun a short smoke job with CUDA_LAUNCH_BLOCKING=1 if the failing operation is unclear.",
        ),
    ),
    FailurePattern(
        "nan_or_inf",
        "fatal",
        "optimization",
        re.compile(r"(\bnan\b|\binf\b|non-finite|overflow)", re.I),
        "Training produced non-finite values.",
        (
            "Lower learning rate or disable unstable mixed precision for a short repro.",
            "Check the latest metrics and data batch around the first non-finite loss.",
        ),
    ),
    FailurePattern(
        "dataloader",
        "fatal",
        "data-loading",
        re.compile(r"(DataLoader worker.*(killed|exited)|bus error|shared memory|BrokenPipeError|EOFError)", re.I),
        "DataLoader or shared-memory worker failed.",
        (
            "Reduce dataloader workers/prefetching if the config exposes it.",
            "Check dataset files and local shared-memory limits.",
        ),
    ),
    FailurePattern(
        "port_in_use",
        "fatal",
        "evaluation-server",
        re.compile(r"(Address already in use|bind.*failed|port .* already in use)", re.I),
        "Evaluation server port is already occupied.",
        (
            "Choose another --port or stop the process using the current port.",
            "Use interaction list/stop to close stale tmux sessions.",
        ),
    ),
    FailurePattern(
        "connection_refused",
        "fatal",
        "evaluation-client",
        re.compile(r"(Connection refused|Failed to establish a new connection|websocket.*refused|server.*not ready)", re.I),
        "Evaluation client could not reach the policy server.",
        (
            "Increase --client-delay or inspect the server log first.",
            "Check that server/client use the same host and port.",
        ),
    ),
    FailurePattern(
        "state_action_dim_mismatch",
        "fatal",
        "model-input",
        re.compile(r"(mat1 and mat2 shapes cannot be multiplied|state_dim|action_dim|Action dim mismatch|State.*dim.*mismatch)", re.I),
        "Model input/action dimensions do not match the selected dataset or checkpoint.",
        (
            "Check dataset action_dim/state_dim/horizon in the generated policy catalog metadata.",
            "For CALVIN eval, verify STARVLA_CALVIN_STATE_DIM and server metadata; current CALVIN checkpoints usually expect 7D state.",
            "Relaunch with the intended --dataset/--action-expert combination so framework.action_model.* dims are generated from the dataset.",
        ),
    ),
    FailurePattern(
        "config_error",
        "fatal",
        "configuration",
        re.compile(r"(omegaconf|ConfigAttributeError|Missing mandatory value|KeyError|ValueError|AssertionError)", re.I),
        "Configuration or argument validation failed.",
        (
            "Inspect the exact key/value in the traceback.",
            "Compare the generated command against the preset config YAML.",
        ),
    ),
    FailurePattern(
        "interrupted",
        "warning",
        "operator",
        re.compile(r"(KeyboardInterrupt|Received signal|SIGTERM|SIGINT|terminated)", re.I),
        "The job was interrupted by a signal or operator action.",
        (
            "If this was not intentional, inspect scheduler/session logs around the interruption time.",
        ),
    ),
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def clean_line(line: str) -> str:
    return strip_ansi(line.rstrip("\n"))


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and METRIC_NUMBER_RE.match(value.strip()):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def compact_value(value: Any) -> str:
    number = safe_float(value)
    if number is not None:
        if abs(number) >= 1000 or (0 < abs(number) < 0.001):
            return f"{number:.3e}"
        return f"{number:.6g}"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= 96 else text[:93] + "..."


def important_metrics(metrics: dict[str, Any], limit: int = 16) -> list[tuple[str, Any]]:
    if not metrics:
        return []
    priority_terms = ("loss", "mse", "lr", "grad", "epoch", "timing", "reward", "accuracy", "acc")
    ordered: list[tuple[str, Any]] = []
    for key in metrics:
        key_lower = key.lower()
        if any(term in key_lower for term in priority_terms):
            ordered.append((key, metrics[key]))
    for key in metrics:
        if len(ordered) >= limit:
            break
        if key not in {item[0] for item in ordered}:
            ordered.append((key, metrics[key]))
    return ordered[:limit]


def extract_last_traceback(lines: Iterable[str], limit: int = 35) -> list[str]:
    cleaned = [clean_line(line) for line in lines]
    start = -1
    for index, line in enumerate(cleaned):
        if "Traceback (most recent call last):" in line:
            start = index
    if start < 0:
        return []
    traceback = cleaned[start:]
    if len(traceback) > limit:
        traceback = traceback[-limit:]
    return traceback


def infer_stage(lines: Iterable[str], latest_event: dict[str, Any] | None = None) -> str:
    event = latest_event or {}
    if event.get("event") == "phase" and event.get("detail"):
        return str(event["detail"])
    if event.get("event") == "failed" and event.get("detail"):
        return "failed-command"

    joined = "\n".join(clean_line(line).lower() for line in lines)
    checks = [
        ("checkpoint-save", ("model checkpoint saved", "full training state saved", "topk_checkpoints")),
        ("resume", ("resumed from checkpoint", "loading checkpoint", "accelerate_state/latest")),
        ("training", ("step ", "loss:", "metrics:", "epoch")),
        ("data-loading", ("dataloader", "dataset", "data_mix", "lerobot")),
        ("model-init", ("loading model", "from_pretrained", "action head", "frozen modules")),
        ("accelerate-launch", ("accelerate launch", "distributed_type", "deepspeed")),
        ("environment", ("source ", "activate_starvla", "cuda_visible_devices")),
    ]
    for stage, needles in checks:
        if any(needle in joined for needle in needles):
            return stage
    if event.get("event"):
        return str(event.get("event"))
    return "unknown"


def diagnose_log(lines: Iterable[str], latest_event: dict[str, Any] | None = None) -> dict[str, Any]:
    cleaned = [clean_line(line) for line in lines if clean_line(line)]
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern in FAILURE_PATTERNS:
        for line in reversed(cleaned):
            if pattern.pattern.search(line):
                if pattern.code in seen:
                    break
                seen.add(pattern.code)
                issues.append(
                    {
                        "code": pattern.code,
                        "severity": pattern.severity,
                        "stage": pattern.stage,
                        "headline": pattern.headline,
                        "evidence": line[-500:],
                        "hints": list(pattern.hints),
                    }
                )
                break

    event = latest_event or {}
    if event.get("event") == "failed" and "runner_failed" not in seen:
        detail = str(event.get("detail") or event.get("command") or "").strip()
        line_no = event.get("line")
        evidence = f"runner exited with status {event.get('status')}"
        if line_no:
            evidence += f" at line {line_no}"
        if detail:
            evidence += f": {detail}"
        runner_issue = {
            "code": "runner_failed",
            "severity": "fatal",
            "stage": "runner",
            "headline": "The tmux runner command failed.",
            "evidence": evidence,
            "hints": ["Inspect the specific root-cause issue and the last traceback."],
        }
        if issues:
            issues.append(runner_issue)
        else:
            issues.insert(0, runner_issue)

    traceback = extract_last_traceback(cleaned)
    severity = "ok"
    if any(issue["severity"] == "fatal" for issue in issues):
        severity = "fatal"
    elif issues:
        severity = "warning"
    return {
        "severity": severity,
        "stage": issues[0]["stage"] if issues else infer_stage(cleaned, latest_event),
        "issues": issues[:5],
        "traceback": traceback,
    }


def read_last_jsonl(path: Path, lines: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except Exception:
        return []
    parsed: list[dict[str, Any]] = []
    for line in raw_lines:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed
