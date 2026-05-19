from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gpu import GPUInfo, command_exists, detect_gpus, gpu_topology
from .paths import CONFIG_ROOT, DEFAULT_ACCELERATE_CONFIG


H200_ACCELERATE_CONFIG = CONFIG_ROOT / "accelerate_h200_zero2.yaml"


@dataclass(frozen=True)
class PerfSettings:
    requested_profile: str
    resolved_profile: str
    accelerate_config: Path
    env_defaults: dict[str, str]
    env_forced: dict[str, str]
    warnings: list[str]
    notes: list[str]


def resolve_perf_settings(
    *,
    profile: str,
    selected_gpus: list[int],
    nccl_debug: str = "WARN",
    nccl_trace: bool = False,
    nccl_socket_ifname: str | None = None,
    nccl_ib_hca: str | None = None,
    nccl_p2p_level: str | None = None,
    env_overrides: list[str] | None = None,
) -> PerfSettings:
    gpus = detect_gpus()
    selected = [gpu for gpu in gpus if gpu.index in set(selected_gpus)] if selected_gpus else []
    topology = gpu_topology().get("output", "")
    has_nvlink = _topology_has_nvlink(str(topology))
    h200_count = sum(1 for gpu in selected if gpu.is_h200)
    resolved = _resolve_profile(profile, selected_gpus, h200_count, has_nvlink)

    env_defaults = {} if resolved == "none" else _base_env(nccl_debug)
    notes: list[str] = []
    warnings: list[str] = []
    accelerate_config = DEFAULT_ACCELERATE_CONFIG

    if resolved == "h200_8gpu":
        accelerate_config = H200_ACCELERATE_CONFIG
        env_defaults.update(
            {
                "TORCH_NCCL_HIGH_PRIORITY": "1",
                "TORCH_NCCL_ENABLE_MONITORING": "1",
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
                "TORCH_NCCL_BLOCKING_WAIT": "0",
                "NCCL_BLOCKING_WAIT": "0",
                "NCCL_P2P_DISABLE": "0",
                "NCCL_NVLS_ENABLE": "2",
                "NCCL_NETDEVS_POLICY": "AUTO",
                "NCCL_SET_THREAD_NAME": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        if has_nvlink:
            env_defaults["NCCL_P2P_LEVEL"] = "NVL"
            notes.append("NVLink topology detected; NCCL_P2P_LEVEL defaults to NVL for intra-node GPU P2P.")
        else:
            warnings.append("Selected H200 profile but NVLink was not visible in nvidia-smi topo output; NCCL_P2P_LEVEL is left to NCCL auto-selection.")
        if len(selected_gpus) != 8:
            warnings.append(f"H200 profile is tuned for 8 GPUs, but {len(selected_gpus)} GPU(s) were selected.")
        if selected and h200_count != len(selected):
            warnings.append("Not every selected GPU reports as H200; verify --gpus selection before launch.")
        notes.append("Using ZeRO-2 h200 config with larger communication buckets and less DeepSpeed print noise.")
    elif resolved == "balanced":
        env_defaults.update(
            {
                "TORCH_NCCL_HIGH_PRIORITY": "1",
                "TORCH_NCCL_ENABLE_MONITORING": "1",
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
                "TORCH_NCCL_BLOCKING_WAIT": "0",
                "NCCL_BLOCKING_WAIT": "0",
                "NCCL_SET_THREAD_NAME": "1",
            }
        )
        notes.append("Using balanced NCCL defaults; topology-sensitive transport choices are left to NCCL.")
    elif resolved == "debug":
        env_defaults.update(
            {
                "TORCH_NCCL_HIGH_PRIORITY": "1",
                "TORCH_NCCL_ENABLE_MONITORING": "1",
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "NCCL_BLOCKING_WAIT": "1",
                "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,COLL",
                "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
                "TORCH_NCCL_DESYNC_DEBUG": "1",
                "TORCH_NCCL_NAN_CHECK": "1",
            }
        )
        notes.append("Debug profile favors root-cause visibility over maximum throughput.")

    if nccl_trace:
        env_defaults["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        env_defaults["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "20000"
    if nccl_socket_ifname:
        env_defaults["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
    if nccl_ib_hca:
        env_defaults["NCCL_IB_HCA"] = nccl_ib_hca
    if nccl_p2p_level:
        env_defaults["NCCL_P2P_LEVEL"] = nccl_p2p_level

    env_forced = parse_env_overrides(env_overrides or [])
    if nccl_debug != "WARN":
        env_forced["NCCL_DEBUG"] = nccl_debug
    if resolved == "h200_8gpu" and not accelerate_config.exists():
        warnings.append(f"H200 accelerate config is missing: {accelerate_config}; falling back to {DEFAULT_ACCELERATE_CONFIG}")
        accelerate_config = DEFAULT_ACCELERATE_CONFIG
    return PerfSettings(
        requested_profile=profile,
        resolved_profile=resolved,
        accelerate_config=accelerate_config,
        env_defaults=env_defaults,
        env_forced=env_forced,
        warnings=warnings,
        notes=notes,
    )


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --nccl-env value {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key!r}")
        overrides[key] = value
    return overrides


def perf_preflight(selected_gpus: list[int], settings: PerfSettings) -> list[list[Any]]:
    gpus = detect_gpus()
    selected = [gpu for gpu in gpus if gpu.index in set(selected_gpus)] if selected_gpus else []
    h200_count = sum(1 for gpu in selected if gpu.is_h200)
    low_free = [gpu.index for gpu in selected if gpu.memory_free_mb < 120_000]
    busy = [gpu.index for gpu in selected if gpu.util_percent > 10]
    power_limited = [
        gpu.index
        for gpu in selected
        if gpu.power_limit_w is not None and gpu.power_w is not None and gpu.power_percent is not None and gpu.power_percent < 15
    ]
    topo = gpu_topology()
    topo_text = str(topo.get("output", ""))
    topo_ok = topo.get("returncode") == 0 or not selected_gpus
    has_all_reduce = command_exists("all_reduce_perf")
    has_p2p = command_exists("p2pBandwidthLatencyTest")
    has_nvbandwidth = command_exists("nvbandwidth")
    return [
        ["perf_profile", settings.resolved_profile, "; ".join(settings.notes) or "-"],
        ["accelerate_config", settings.accelerate_config.exists(), str(settings.accelerate_config)],
        ["selected_gpus", len(selected_gpus), ",".join(map(str, selected_gpus)) or "-"],
        ["selected_h200", h200_count, "all selected GPUs should be H200 for the H200 profile"],
        ["free_memory_check", not low_free, f"GPUs below 120GB free before launch: {low_free or '-'}"],
        ["prelaunch_gpu_busy", not busy, f"GPUs above 10% util before launch: {busy or '-'}"],
        ["prelaunch_power_idle", True, f"low power before launch is normal if idle: {power_limited or '-'}"],
        ["topology_visible", topo_ok, "NVLink present" if _topology_has_nvlink(topo_text) else "NVLink not detected in nvidia-smi topo"],
        ["all_reduce_perf", has_all_reduce, "available" if has_all_reduce else "install/build NVIDIA nccl-tests for bandwidth validation"],
        ["p2pBandwidthLatencyTest", True, "available" if has_p2p else "optional CUDA sample not found; all_reduce_perf remains the required bandwidth smoke test"],
        ["nvbandwidth", True, "available" if has_nvbandwidth else "optional NVIDIA nvbandwidth not found; install it only for deeper memory/interconnect benchmarking"],
    ]


def _base_env(nccl_debug: str) -> dict[str, str]:
    env = {
        "NCCL_DEBUG": nccl_debug,
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
        "NCCL_SOCKET_RETRY_CNT": "34",
        "NCCL_SOCKET_RETRY_SLEEP_MSEC": "100",
    }
    if nccl_debug != "WARN":
        env["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH"
    return env


def _resolve_profile(profile: str, selected_gpus: list[int], h200_count: int, has_nvlink: bool) -> str:
    if profile != "auto":
        return profile
    if len(selected_gpus) >= 8 and h200_count >= 8:
        return "h200_8gpu"
    if has_nvlink and len(selected_gpus) >= 4:
        return "balanced"
    return "balanced"


def _topology_has_nvlink(text: str) -> bool:
    return bool(re.search(r"\bNV\d+\b|\bNVL\b", text))
