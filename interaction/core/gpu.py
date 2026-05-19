from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    util_percent: int
    temperature_c: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    pstate: str | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    compute_mode: str | None = None
    persistence_mode: str | None = None

    @property
    def memory_free_mb(self) -> int:
        return max(self.memory_total_mb - self.memory_used_mb, 0)

    @property
    def memory_used_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return min(max(self.memory_used_mb * 100 / self.memory_total_mb, 0.0), 100.0)

    @property
    def power_percent(self) -> float | None:
        if self.power_w is None or self.power_limit_w in {None, 0}:
            return None
        return min(max(self.power_w * 100 / self.power_limit_w, 0.0), 200.0)

    @property
    def is_h200(self) -> bool:
        return "H200" in self.name.upper()


@dataclass(frozen=True)
class GPUProcessInfo:
    gpu_uuid: str
    gpu_index: int | None
    pid: int
    process_name: str
    used_memory_mb: int


def nvidia_smi_binary() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/opt/nvidia/bin/nvidia-smi",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def detect_gpus() -> list[GPUInfo]:
    extended_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,pstate,clocks.sm,clocks.mem,compute_mode,persistence_mode",
        "--format=csv,noheader,nounits",
    ]
    basic_cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,pstate",
        "--format=csv,noheader,nounits",
    ]
    proc = _run_nvidia_smi(extended_cmd)
    if proc is None:
        proc = _run_nvidia_smi(basic_cmd)
    if proc is None:
        return []

    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in {9, 14}:
            continue
        try:
            base = dict(
                index=int(float(parts[0])),
                uuid=parts[1],
                name=parts[2],
                memory_total_mb=int(float(parts[3])),
                memory_used_mb=int(float(parts[4])),
                util_percent=int(float(parts[5])),
                temperature_c=_int_or_none(parts[6]),
                power_w=_float_or_none(parts[7]),
            )
            if len(parts) == 14:
                gpus.append(
                    GPUInfo(
                        **base,
                        power_limit_w=_float_or_none(parts[8]),
                        pstate=parts[9] or None,
                        sm_clock_mhz=_int_or_none(parts[10]),
                        mem_clock_mhz=_int_or_none(parts[11]),
                        compute_mode=parts[12] or None,
                        persistence_mode=parts[13] or None,
                    )
                )
            else:
                gpus.append(GPUInfo(**base, pstate=parts[8] or None))
        except ValueError:
            continue
    return gpus


def _run_nvidia_smi(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    binary = nvidia_smi_binary()
    if not binary:
        return None
    command = [binary, *cmd[1:]]
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def command_exists(command: str) -> bool:
    if command == "nvidia-smi":
        return nvidia_smi_binary() is not None
    return shutil.which(command) is not None


def run_text_command(command: list[str], timeout: int = 10) -> tuple[int, str]:
    if command and command[0] == "nvidia-smi":
        binary = nvidia_smi_binary()
        if not binary:
            return 127, "nvidia-smi not found"
        command = [binary, *command[1:]]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        return 127, repr(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + (exc.stderr or "") + "\nTIMEOUT"
    return proc.returncode, proc.stdout + proc.stderr


def gpu_topology() -> dict[str, str | int]:
    rc, out = run_text_command(["nvidia-smi", "topo", "-m"], timeout=10)
    return {"returncode": rc, "output": out[-12000:]}


def gpu_nvlink_status() -> dict[str, str | int]:
    rc, out = run_text_command(["nvidia-smi", "nvlink", "-s"], timeout=10)
    return {"returncode": rc, "output": out[-12000:]}


def detect_compute_processes() -> list[GPUProcessInfo]:
    gpus = detect_gpus()
    uuid_to_index = {gpu.uuid: gpu.index for gpu in gpus}
    proc = _run_nvidia_smi(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None:
        return []
    processes: list[GPUProcessInfo] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            used = int(float(parts[3]))
            pid = int(float(parts[1]))
        except ValueError:
            continue
        processes.append(
            GPUProcessInfo(
                gpu_uuid=parts[0],
                gpu_index=uuid_to_index.get(parts[0]),
                pid=pid,
                process_name=parts[2],
                used_memory_mb=used,
            )
        )
    return processes


def visible_gpu_indexes() -> list[int]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS")
    if not value:
        return [gpu.index for gpu in detect_gpus()]
    indexes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indexes.append(int(item))
        except ValueError:
            continue
    return indexes


def _filter_visible(gpus: Iterable[GPUInfo]) -> list[GPUInfo]:
    visible = visible_gpu_indexes()
    if not visible:
        return list(gpus)
    allowed = set(visible)
    return [gpu for gpu in gpus if gpu.index in allowed]


def select_gpus(request: str | None, num_gpus: int | None = None, min_free_mb: int = 60000) -> list[int]:
    gpus = detect_gpus()
    if request and request not in {"auto", "all"}:
        return [int(part.strip()) for part in request.split(",") if part.strip()]

    if not gpus:
        return []

    visible_gpus = _filter_visible(gpus)
    if visible_gpus:
        gpus = visible_gpus

    if request == "all" and num_gpus is None:
        return [gpu.index for gpu in gpus]

    candidates = [gpu for gpu in gpus if gpu.memory_free_mb >= min_free_mb]
    if not candidates:
        candidates = gpus
    candidates.sort(key=lambda gpu: (gpu.is_h200, gpu.memory_free_mb, gpu.memory_total_mb, -gpu.util_percent), reverse=True)
    count = num_gpus or len(candidates)
    return [gpu.index for gpu in candidates[:count]]


def format_gpu_table(gpus: list[GPUInfo]) -> list[dict[str, str]]:
    rows = []
    for gpu in gpus:
        power = "-" if gpu.power_w is None else f"{gpu.power_w:.0f}W"
        if gpu.power_limit_w:
            power += f"/{gpu.power_limit_w:.0f}W"
        power_pct = "-" if gpu.power_percent is None else f"{gpu.power_percent:.0f}%"
        rows.append(
            {
                "id": str(gpu.index),
                "name": gpu.name,
                "h200": "yes" if gpu.is_h200 else "no",
                "used": f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB ({gpu.memory_used_percent:.0f}%)",
                "free": f"{gpu.memory_free_mb} MB",
                "util": f"{gpu.util_percent}%",
                "temp": "-" if gpu.temperature_c is None else f"{gpu.temperature_c}C",
                "power": power,
                "power_pct": power_pct,
                "sm_clock": "-" if gpu.sm_clock_mhz is None else f"{gpu.sm_clock_mhz}MHz",
                "mem_clock": "-" if gpu.mem_clock_mhz is None else f"{gpu.mem_clock_mhz}MHz",
                "pstate": gpu.pstate or "-",
                "compute_mode": gpu.compute_mode or "-",
                "persistence": gpu.persistence_mode or "-",
            }
        )
    return rows
