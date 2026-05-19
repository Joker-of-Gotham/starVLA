from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import dataset_name, experiment_group_name, load_catalog, model_path
from .gpu import select_gpus
from .perf import PerfSettings, resolve_perf_settings
from .paths import (
    ACTIVATE_SCRIPT,
    DEFAULT_ACCELERATE_CONFIG,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_PYTHON,
    DESIRED_RUNS_ROOT,
    FALLBACK_RUNS_ROOT,
    REPO_ROOT,
    artifact_roots_report,
    ensure_runtime_dirs,
)
from .throughput import as_meta as throughput_as_meta
from .throughput import resolve_throughput_settings


def q(value: str | Path | int | float | bool) -> str:
    return shlex.quote(str(value))


def _export_default(key: str, value: str) -> str:
    return f"if [[ -z \"${{{key}:-}}\" ]]; then export {key}={q(value)}; fi"


def _export_force(key: str, value: str) -> str:
    return f"export {key}={q(value)}"


def _tracked_accelerate_launch_lines(args: list[str]) -> list[str]:
    launch_cmd = "exec " + " ".join(q(item) for item in args)
    return [
        "_starvla_accelerate_status=0",
        "if command -v setsid >/dev/null 2>&1; then",
        f"  setsid bash -lc {q(launch_cmd)} &",
        "else",
        f"  bash -lc {q(launch_cmd)} &",
        "fi",
        "STARVLA_ACCELERATE_PID=$!",
        'STARVLA_ACCELERATE_PGID=""',
        'for _starvla_pgid_try in 1 2 3 4 5; do',
        '  STARVLA_ACCELERATE_PGID="$(ps -o pgid= -p "$STARVLA_ACCELERATE_PID" 2>/dev/null | tr -d " " || true)"',
        '  [[ -n "${STARVLA_ACCELERATE_PGID:-}" ]] && break',
        "  sleep 0.2",
        "done",
        'if [[ -n "${STARVLA_JOB_DIR:-}" ]]; then',
        '  echo "$STARVLA_ACCELERATE_PID" > "$STARVLA_JOB_DIR/${STARVLA_WINDOW_NAME}.accelerate.pid"',
        '  if [[ -n "${STARVLA_ACCELERATE_PGID:-}" ]]; then',
        '    echo "$STARVLA_ACCELERATE_PGID" > "$STARVLA_JOB_DIR/${STARVLA_WINDOW_NAME}.accelerate.pgid"',
        "  fi",
        "fi",
        "set +e",
        'wait "$STARVLA_ACCELERATE_PID"',
        "_starvla_accelerate_status=$?",
        "set -e",
        'if [[ "$_starvla_accelerate_status" -ne 0 ]]; then',
        '  echo "[starvla-error] accelerate exited with status=${_starvla_accelerate_status}"',
        '  if [[ -n "${STARVLA_ACCELERATE_PGID:-}" && "${STARVLA_ACCELERATE_PGID}" != "${STARVLA_RUNNER_PGID:-}" ]]; then',
        '    kill -TERM "-${STARVLA_ACCELERATE_PGID}" 2>/dev/null || true',
        "    sleep 3",
        '    kill -KILL "-${STARVLA_ACCELERATE_PGID}" 2>/dev/null || true',
        "  fi",
        '  if declare -F _starvla_event >/dev/null; then _starvla_event failed "$_starvla_accelerate_status" "accelerate_launch" "${BASH_LINENO[0]:-}"; fi',
        '  exit "$_starvla_accelerate_status"',
        "fi",
    ]


@dataclass
class TrainLaunch:
    preset: str
    framework: str
    model: str
    gpus: list[int]
    run_id: str
    max_train_steps: int | None = None
    vla_batch_size: int | None = None
    vlm_batch_size: int | None = None
    data_root: str | None = None
    data_mix: str | None = None
    vlm_dataset: str | None = None
    run_root_dir: str | None = None
    experiment_group: str | None = None
    resume: bool = False
    resume_from_checkpoint: str | None = None
    resume_mode: str = "auto"
    save_full_state: bool = True
    keep_top_k: int = 3
    freeze_modules: str | None = None
    mixed_precision: str = "bf16"
    num_machines: int = 1
    machine_rank: int = 0
    main_process_ip: str | None = None
    main_process_port: int | None = None
    disable_wandb: bool = True
    perf_profile: str = "auto"
    nccl_debug: str = "WARN"
    nccl_trace: bool = False
    nccl_socket_ifname: str | None = None
    nccl_ib_hca: str | None = None
    nccl_p2p_level: str | None = None
    nccl_env: list[str] = field(default_factory=list)
    throughput_profile: str = "auto"
    dataloader_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_pin_memory: bool | None = None
    dataloader_persistent_workers: bool | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class EvalLaunch:
    preset: str
    ckpt: str
    server_gpu: str = "0"
    client_gpu: str = "0"
    port: int | None = None
    start_client: bool = True
    task: str | None = None
    trials: int | None = None
    max_steps_per_task: int | None = None
    parallel_workers: int | None = None
    replicas_per_gpu: int | None = None
    eval_throughput_profile: str = "auto"
    calvin_render_backend: str = "auto"
    dataset_path: str | None = None
    eval_log_dir: str | None = None
    extra_args: list[str] = field(default_factory=list)


def _common_header(log_file: Path, job_dir: Path | None = None) -> list[str]:
    job_lines: list[str] = []
    if job_dir is not None:
        job_lines = [
            f"export STARVLA_JOB_DIR={q(job_dir)}",
            f"export STARVLA_EVENTS_FILE={q(job_dir / 'events.jsonl')}",
            f"export STARVLA_LOG_FILE={q(log_file)}",
            f"export STARVLA_WINDOW_NAME={q(log_file.stem)}",
            "mkdir -p \"$STARVLA_JOB_DIR\"",
            "echo \"$$\" > \"$STARVLA_JOB_DIR/${STARVLA_WINDOW_NAME}.pid\"",
            "export STARVLA_RUNNER_PID=\"$$\"",
            "export STARVLA_RUNNER_PGID=\"$(ps -o pgid= -p \"$$\" 2>/dev/null | tr -d ' ' || true)\"",
            "_starvla_event() {",
            "  local event_name=\"${1:-event}\"",
            "  local status=\"${2:-0}\"",
            "  local detail=\"${3:-}\"",
            "  local line=\"${4:-}\"",
            "  local python_bin=\"\"",
            "  python_bin=\"$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)\"",
            "  if [[ -z \"$python_bin\" ]]; then",
            "    printf '{\"event\":\"%s\",\"status\":\"%s\",\"detail\":\"%s\",\"line\":\"%s\"}\\n' \"$event_name\" \"$status\" \"$detail\" \"$line\" >> \"$STARVLA_EVENTS_FILE\" || true",
            "    return 0",
            "  fi",
            "  \"$python_bin\" - \"$event_name\" \"$status\" \"$detail\" \"$line\" <<'PY'",
            "import json, os, subprocess, sys, time",
            "event = {",
            "  'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
            "  'event': sys.argv[1],",
            "  'status': sys.argv[2],",
            "  'detail': sys.argv[3],",
            "  'line': sys.argv[4],",
            "  'pid': os.getpid(),",
            "  'window': os.environ.get('STARVLA_WINDOW_NAME'),",
            "  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),",
            "}",
            "try:",
            "  out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,pstate,temperature.gpu,clocks.sm,clocks.mem', '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)",
            "  event['gpu_snapshot'] = out.stdout.splitlines()",
            "except Exception as exc:",
            "  event['gpu_snapshot_error'] = repr(exc)",
            "with open(os.environ['STARVLA_EVENTS_FILE'], 'a', encoding='utf-8') as f:",
            "  f.write(json.dumps(event, sort_keys=True) + '\\n')",
            "PY",
            "}",
            "_starvla_phase() {",
            "  local detail=\"$*\"",
            "  echo \"[starvla-phase] ${detail}\"",
            "  _starvla_event phase 0 \"$detail\" \"${BASH_LINENO[0]:-}\"",
            "}",
            "_starvla_heartbeat() { while true; do _starvla_event heartbeat 0 \"alive\" \"${BASH_LINENO[0]:-}\"; sleep \"${STARVLA_HEARTBEAT_SECONDS:-60}\"; done; }",
            "_starvla_cleanup() {",
            "  local status=$?",
            "  set +e",
            "  kill \"${STARVLA_HEARTBEAT_PID:-}\" 2>/dev/null || true",
            "  wait \"${STARVLA_HEARTBEAT_PID:-}\" 2>/dev/null || true",
            "  if [[ -n \"${STARVLA_ACCELERATE_PGID:-}\" && \"${STARVLA_ACCELERATE_PGID}\" != \"${STARVLA_RUNNER_PGID:-}\" ]]; then",
            "    kill -TERM \"-${STARVLA_ACCELERATE_PGID}\" 2>/dev/null || true",
            "    sleep 1",
            "    kill -KILL \"-${STARVLA_ACCELERATE_PGID}\" 2>/dev/null || true",
            "  fi",
            "  if [[ -n \"${STARVLA_ACCELERATE_PID:-}\" ]]; then",
            "    kill -TERM \"${STARVLA_ACCELERATE_PID}\" 2>/dev/null || true",
            "    sleep 1",
            "    kill -KILL \"${STARVLA_ACCELERATE_PID}\" 2>/dev/null || true",
            "  fi",
            "  _starvla_event finished \"$status\" \"exit\" \"${BASH_LINENO[0]:-}\"",
            "  exit \"$status\"",
            "}",
            "_starvla_error_trap() {",
            "  local status=$?",
            "  local failed_line=\"${BASH_LINENO[0]:-}\"",
            "  local failed_command=\"${BASH_COMMAND:-unknown}\"",
            "  set +e",
            "  echo \"[starvla-error] status=${status} line=${failed_line} command=${failed_command}\"",
            "  _starvla_event failed \"$status\" \"$failed_command\" \"$failed_line\"",
            "  exit \"$status\"",
            "}",
            "trap _starvla_error_trap ERR",
            "trap _starvla_cleanup EXIT",
            "_starvla_event started 0 bootstrap \"${BASH_LINENO[0]:-}\"",
            "_starvla_heartbeat &",
            "STARVLA_HEARTBEAT_PID=$!",
            "_starvla_phase bootstrap",
        ]
    snapshot_lines: list[str] = []
    if job_dir is not None:
        snapshot_lines = [
            "_starvla_phase snapshot_environment",
            "python - <<'PY'",
            "import json, os, platform, subprocess, time",
            "job_dir = os.environ['STARVLA_JOB_DIR']",
            "window = os.environ.get('STARVLA_WINDOW_NAME', 'main')",
            "snapshot = {",
            "  'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
            "  'window': window,",
            "  'python': os.popen('command -v python').read().strip(),",
            "  'hostname': platform.node(),",
            "  'cwd': os.getcwd(),",
            "  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),",
            "  'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'),",
            "  'tmux': os.environ.get('TMUX'),",
            "  'proxy': {k: os.environ.get(k) for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']},",
            "  'nccl_env': {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith('NCCL') or k.startswith('TORCH_NCCL')},",
            "  'perf_env': {k: os.environ.get(k) for k in ['STARVLA_PERF_PROFILE','CUDA_DEVICE_MAX_CONNECTIONS','PYTORCH_CUDA_ALLOC_CONF']},",
            "}",
            "try:",
            "  snapshot['nvidia_smi_l'] = subprocess.run(['nvidia-smi', '-L'], text=True, capture_output=True, timeout=10).stdout.splitlines()",
            "  snapshot['nvidia_smi_topo'] = subprocess.run(['nvidia-smi', 'topo', '-m'], text=True, capture_output=True, timeout=10).stdout.splitlines()",
            "except Exception as exc:",
            "  snapshot['nvidia_smi_error'] = repr(exc)",
            "with open(os.path.join(job_dir, f'environment.{window}.json'), 'w', encoding='utf-8') as f:",
            "  json.dump(snapshot, f, indent=2, sort_keys=True)",
            "if window in {'main', 'server', 'selftest'}:",
            "  with open(os.path.join(job_dir, 'environment.json'), 'w', encoding='utf-8') as f:",
            "    json.dump(snapshot, f, indent=2, sort_keys=True)",
            "PY",
            "_starvla_phase ready",
        ]
    if job_dir is not None:
        redirect_line = f"exec > >(tee -a {q(log_file)} {q(job_dir / 'logs' / 'all.log')}) 2>&1"
    else:
        redirect_line = f"exec > >(tee -a {q(log_file)}) 2>&1"
    return [
        "#!/usr/bin/env bash",
        "set -Eeuo pipefail",
        f"cd {q(REPO_ROOT)}",
        redirect_line,
        "echo \"[starvla-interaction] started at $(date -Is)\"",
    ] + job_lines + [
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy",
        "unset DEBUG",
        "export STARVLA_ENABLE_DEBUGPY=${STARVLA_ENABLE_DEBUGPY:-0}",
        "unset PYDEVD_LOAD_VALUES_ASYNC PYDEVD_USE_FRAME_EVAL DEBUGPY_LAUNCHER_PORT",
        "export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,::1}",
        "if declare -F _starvla_phase >/dev/null; then _starvla_phase activate_environment; fi",
        f"source {q(ACTIVATE_SCRIPT)}",
        "export TOKENIZERS_PARALLELISM=false",
        "export ACCELERATE_LOG_LEVEL=${ACCELERATE_LOG_LEVEL:-info}",
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}",
        "export TORCH_NCCL_USE_COMM_NONBLOCKING=${TORCH_NCCL_USE_COMM_NONBLOCKING:-1}",
        "export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-0}",
        "export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}",
        "export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}",
        "export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}",
        "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}",
        "export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}",
        "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
    ] + snapshot_lines


def _accelerate_prefix(
    num_processes: int,
    config_file: str | Path = DEFAULT_ACCELERATE_CONFIG,
    mixed_precision: str = "bf16",
    num_machines: int = 1,
    machine_rank: int = 0,
    main_process_ip: str | None = None,
    main_process_port: int | None = None,
) -> list[str]:
    args = [
        "accelerate",
        "launch",
        "--config_file",
        str(config_file),
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        str(machine_rank),
    ]
    if mixed_precision and mixed_precision != "config":
        args += ["--mixed_precision", mixed_precision]
    if main_process_ip:
        args += ["--main_process_ip", main_process_ip]
    if main_process_port:
        args += ["--main_process_port", str(main_process_port)]
    return args


def build_train_command(spec: TrainLaunch) -> tuple[str, dict[str, Any]]:
    catalog = load_catalog()
    preset = dict(catalog["train_presets"][spec.preset])
    mode = preset["mode"]
    trainer_script = preset["trainer_script"]

    model = model_path(spec.model)
    data_root = spec.data_root if spec.data_root is not None else preset.get("data_root")
    data_mix = spec.data_mix if spec.data_mix is not None else preset.get("data_mix")
    vlm_dataset = spec.vlm_dataset if spec.vlm_dataset is not None else preset.get("vlm_dataset")
    dataset = dataset_name(spec.preset, data_mix=data_mix, vlm_dataset=vlm_dataset)
    experiment_group = spec.experiment_group or experiment_group_name(spec.model, spec.framework, dataset)
    checkpoint_root = spec.run_root_dir if spec.run_root_dir is not None else preset.get("run_root_dir", str(DEFAULT_CHECKPOINT_ROOT))
    run_root_dir = str(Path(checkpoint_root) / experiment_group)
    freeze_modules = spec.freeze_modules if spec.freeze_modules is not None else _default_freeze_modules(spec.framework, spec.model)
    max_train_steps = spec.max_train_steps if spec.max_train_steps is not None else int(preset["max_train_steps"])
    vla_batch = spec.vla_batch_size if spec.vla_batch_size is not None else int(preset.get("vla_batch_size", 1))
    vlm_batch = spec.vlm_batch_size if spec.vlm_batch_size is not None else int(preset.get("vlm_batch_size", 1))
    num_processes = max(len(spec.gpus), 1)
    throughput = resolve_throughput_settings(
        profile=spec.throughput_profile,
        selected_gpus=spec.gpus,
        model_key=spec.model,
        framework=spec.framework,
        mode=mode,
        preset_vla_batch=vla_batch,
        preset_vlm_batch=vlm_batch,
        explicit_vla_batch=spec.vla_batch_size,
        explicit_vlm_batch=spec.vlm_batch_size,
        explicit_workers=spec.dataloader_workers,
        explicit_prefetch=spec.dataloader_prefetch_factor,
        explicit_pin_memory=spec.dataloader_pin_memory,
        explicit_persistent_workers=spec.dataloader_persistent_workers,
    )
    vla_batch = throughput.vla_batch_size if throughput.vla_batch_size is not None else vla_batch
    vlm_batch = throughput.vlm_batch_size if throughput.vlm_batch_size is not None else vlm_batch
    perf = resolve_perf_settings(
        profile=spec.perf_profile,
        selected_gpus=spec.gpus,
        nccl_debug=spec.nccl_debug,
        nccl_trace=spec.nccl_trace,
        nccl_socket_ifname=spec.nccl_socket_ifname,
        nccl_ib_hca=spec.nccl_ib_hca,
        nccl_p2p_level=spec.nccl_p2p_level,
        env_overrides=spec.nccl_env,
    )

    args = _accelerate_prefix(
        num_processes,
        config_file=perf.accelerate_config,
        mixed_precision=spec.mixed_precision,
        num_machines=spec.num_machines,
        machine_rank=spec.machine_rank,
        main_process_ip=spec.main_process_ip,
        main_process_port=spec.main_process_port,
    )
    args += [
        trainer_script,
        "--config_yaml",
        preset["config_yaml"],
        "--framework.name",
        spec.framework,
        "--framework.qwenvl.base_vlm",
        model,
        "--trainer.freeze_modules",
        freeze_modules,
        "--trainer.max_train_steps",
        str(max_train_steps),
        "--trainer.save_interval",
        str(preset["save_interval"]),
        "--trainer.logging_frequency",
        str(preset["logging_frequency"]),
        "--trainer.eval_interval",
        str(preset["eval_interval"]),
        "--run_root_dir",
        run_root_dir,
        "--run_id",
        spec.run_id,
        "--wandb_project",
        preset.get("wandb_project", "starVLA"),
        "--wandb_entity",
        preset.get("wandb_entity", "your_wandb_entity"),
        "--trainer.save_full_state",
        str(bool(spec.save_full_state)),
        "--trainer.resume_mode",
        spec.resume_mode,
        "--trainer.keep_top_k",
        str(spec.keep_top_k),
    ]

    if spec.resume:
        args += ["--trainer.is_resume", "True"]
    if spec.resume_from_checkpoint:
        args += ["--trainer.resume_from_checkpoint", spec.resume_from_checkpoint]

    if mode in {"vla", "cotrain"}:
        if data_root:
            args += ["--datasets.vla_data.data_root_dir", data_root]
        if data_mix:
            args += ["--datasets.vla_data.data_mix", data_mix]
        args += [
            "--datasets.vla_data.per_device_batch_size",
            str(vla_batch),
            "--datasets.vla_data.video_backend",
            "torchvision_av",
        ]
        if throughput.dataloader_workers is not None:
            args += ["--datasets.vla_data.num_workers", str(throughput.dataloader_workers)]
        if throughput.prefetch_factor is not None:
            args += ["--datasets.vla_data.prefetch_factor", str(throughput.prefetch_factor)]
        if throughput.pin_memory is not None:
            args += ["--datasets.vla_data.pin_memory", str(bool(throughput.pin_memory))]
        if throughput.persistent_workers is not None:
            args += ["--datasets.vla_data.persistent_workers", str(bool(throughput.persistent_workers))]
    if mode == "cotrain":
        args += ["--datasets.vlm_data.per_device_batch_size", str(vlm_batch)]
        if throughput.dataloader_workers is not None:
            args += ["--datasets.vlm_data.num_workers", str(throughput.dataloader_workers)]
        if throughput.prefetch_factor is not None:
            args += ["--datasets.vlm_data.prefetch_factor", str(throughput.prefetch_factor)]
        if throughput.pin_memory is not None:
            args += ["--datasets.vlm_data.pin_memory", str(bool(throughput.pin_memory))]
        if throughput.persistent_workers is not None:
            args += ["--datasets.vlm_data.persistent_workers", str(bool(throughput.persistent_workers))]
    if mode == "vlm":
        if vlm_dataset:
            args += ["--datasets.vlm_data.dataset_use", vlm_dataset]
        args += ["--datasets.vlm_data.per_device_batch_size", str(vlm_batch)]
        if throughput.dataloader_workers is not None:
            args += ["--datasets.vlm_data.num_workers", str(throughput.dataloader_workers)]
        if throughput.prefetch_factor is not None:
            args += ["--datasets.vlm_data.prefetch_factor", str(throughput.prefetch_factor)]
        if throughput.pin_memory is not None:
            args += ["--datasets.vlm_data.pin_memory", str(bool(throughput.pin_memory))]
        if throughput.persistent_workers is not None:
            args += ["--datasets.vlm_data.persistent_workers", str(bool(throughput.persistent_workers))]

    args += spec.extra_args

    lines = []
    lines.append(f"if declare -F _starvla_phase >/dev/null; then _starvla_phase train_prepare preset={q(spec.preset)} run_id={q(spec.run_id)}; fi")
    if spec.gpus:
        lines.append(f"export CUDA_VISIBLE_DEVICES={q(','.join(map(str, spec.gpus)))}")
    lines.append(f"export STARVLA_PERF_PROFILE={q(perf.resolved_profile)}")
    for key, value in sorted(perf.env_defaults.items()):
        lines.append(_export_default(key, value))
    for key, value in sorted(throughput.env_defaults.items()):
        lines.append(_export_default(key, value))
    for key, value in sorted(perf.env_forced.items()):
        lines.append(_export_force(key, value))
    lines.append("if [[ -n \"${STARVLA_JOB_DIR:-}\" ]]; then export NCCL_TOPO_DUMP_FILE=\"${NCCL_TOPO_DUMP_FILE:-${STARVLA_JOB_DIR}/nccl_topology.xml}\"; fi")
    lines.append("if [[ -n \"${STARVLA_JOB_DIR:-}\" && -n \"${TORCH_NCCL_TRACE_BUFFER_SIZE:-}\" && \"${TORCH_NCCL_TRACE_BUFFER_SIZE}\" != \"0\" ]]; then export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=\"${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-${STARVLA_JOB_DIR}/torch_nccl_debug.json}\"; fi")
    if spec.disable_wandb:
        lines.append("export WANDB_MODE=disabled")
    lines.append("echo \"[starvla-interaction] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}\"")
    lines.append("echo \"[starvla-interaction] perf_profile=${STARVLA_PERF_PROFILE}\"")
    lines.append("echo \"[starvla-interaction] throughput_profile=" + q(throughput.resolved_profile) + "\"")
    lines.append("env | sort | grep -E '^(NCCL|TORCH_NCCL|CUDA_DEVICE_MAX_CONNECTIONS|PYTORCH_CUDA_ALLOC_CONF|STARVLA_PERF_PROFILE|OMP_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|TORCH_ALLOW_TF32_CUBLAS_OVERRIDE|NVIDIA_TF32_OVERRIDE)=' || true")
    lines.append("echo \"[starvla-interaction] command:\"")
    lines.append("printf '%q ' " + " ".join(q(item) for item in args) + "; echo")
    lines.append("if declare -F _starvla_phase >/dev/null; then _starvla_phase accelerate_launch; fi")
    lines.extend(_tracked_accelerate_launch_lines(args))
    lines.append("if declare -F _starvla_phase >/dev/null; then _starvla_phase train_completed; fi")

    meta = {
        "kind": "train",
        "preset": spec.preset,
        "mode": mode,
        "framework": spec.framework,
        "model": model,
        "gpus": spec.gpus,
        "num_processes": num_processes,
        "num_machines": spec.num_machines,
        "mixed_precision": spec.mixed_precision,
        "perf_profile": perf.resolved_profile,
        "perf_profile_requested": perf.requested_profile,
        "perf_warnings": perf.warnings,
        "perf_notes": perf.notes,
        "accelerate_config": str(perf.accelerate_config),
        "nccl_env_defaults": perf.env_defaults,
        "nccl_env_forced": perf.env_forced,
        "run_id": spec.run_id,
        "checkpoint_root": checkpoint_root,
        "experiment_group": experiment_group,
        "dataset": dataset,
        "group_root_dir": run_root_dir,
        "run_root_dir": run_root_dir,
        "output_dir": str(Path(run_root_dir) / spec.run_id),
        "checkpoint_dir": str(Path(run_root_dir) / spec.run_id / "checkpoints"),
        "full_state_dir": str(Path(run_root_dir) / spec.run_id / "accelerate_state"),
        "final_model_dir": str(Path(run_root_dir) / spec.run_id / "final_model"),
        "summary_jsonl": str(Path(run_root_dir) / spec.run_id / "summary.jsonl"),
        "resume": spec.resume,
        "resume_from_checkpoint": spec.resume_from_checkpoint,
        "resume_mode": spec.resume_mode,
        "save_full_state": spec.save_full_state,
        "keep_top_k": spec.keep_top_k,
        "freeze_modules": freeze_modules,
        "max_train_steps": max_train_steps,
        "trainer_script": trainer_script,
        "config_yaml": preset["config_yaml"],
        "data_root": data_root,
        "data_mix": data_mix,
    }
    meta.update(throughput_as_meta(throughput))
    return "\n".join(lines) + "\n", meta


def _default_freeze_modules(framework: str, model_key: str) -> str:
    text = f"{framework} {model_key}".lower()
    if "cosmo" in text or "cosmos" in text:
        return "backbone"
    return ""


def _resolve_eval_server_gpus(request: str | None) -> list[int]:
    text = (request or "0").strip().lower()
    if text in {"", "none", "cpu"}:
        return []
    try:
        return select_gpus(text, num_gpus=None, min_free_mb=8000)
    except Exception:
        if text in {"auto", "all"}:
            return []
        gpus: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                gpus.append(int(part))
            except ValueError:
                continue
        return gpus


def _eval_replicas_per_gpu(profile: str, explicit: int | None, preset: dict[str, Any]) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    profile = (profile or "auto").strip().lower()
    if profile == "conservative":
        return 1
    if profile == "balanced":
        return 2
    if profile == "h200_saturated":
        return int(preset.get("h200_saturated_replicas_per_gpu", 3))
    return int(preset.get("default_replicas_per_gpu", 2))


def _calvin_merge_command(worker_count: int, eval_log_base: str | None) -> str:
    lines = [
        "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_merge_prepare preset=calvin; fi",
        f"export STARVLA_CALVIN_WORKERS={q(worker_count)}",
        "export STARVLA_CALVIN_MERGE_TIMEOUT=${STARVLA_CALVIN_MERGE_TIMEOUT:-604800}",
    ]
    if eval_log_base:
        lines.append(f"export STARVLA_CALVIN_EVAL_LOG_BASE={q(eval_log_base)}")
    else:
        lines.append("unset STARVLA_CALVIN_EVAL_LOG_BASE")
    lines += [
        "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_merge_wait; fi",
        "python - <<'PY'",
        "import json, os, time",
        "from collections import Counter",
        "from pathlib import Path",
        "worker_count = int(os.environ['STARVLA_CALVIN_WORKERS'])",
        "timeout = int(os.environ.get('STARVLA_CALVIN_MERGE_TIMEOUT', '604800'))",
        "job_dir = Path(os.environ.get('STARVLA_JOB_DIR', '.'))",
        "base = os.environ.get('STARVLA_CALVIN_EVAL_LOG_BASE')",
        "base_dir = Path(base) if base else job_dir / 'calvin_eval'",
        "if base:",
        "    dirs = [base_dir / 'workers' / f'{i:03d}' for i in range(worker_count)]",
        "else:",
        "    dirs = [base_dir / 'workers' / f'{i:03d}' for i in range(worker_count)]",
        "files = [path / 'sequence_results.json' for path in dirs]",
        "deadline = time.time() + timeout",
        "last_report = 0.0",
        "while True:",
        "    missing = [str(path) for path in files if not path.exists()]",
        "    if not missing:",
        "        break",
        "    if time.time() > deadline:",
        "        raise TimeoutError('Timed out waiting for CALVIN worker results: ' + ', '.join(missing))",
        "    if time.time() - last_report > 60:",
        "        print(f'[starvla-calvin-merge] waiting for {len(missing)}/{len(files)} worker result files')",
        "        last_report = time.time()",
        "    time.sleep(10)",
        "raw = []",
        "for path in files:",
        "    raw.extend(json.loads(path.read_text()))",
        "raw.sort(key=lambda item: int(item.get('sequence_index', -1)))",
        "if not raw:",
        "    raise RuntimeError('No CALVIN sequence results to merge')",
        "results = [int(item['success_count']) for item in raw]",
        "chain_sr = {str(i): sum(1 for value in results if value >= i) / len(results) for i in range(1, 6)}",
        "cnt_success, cnt_fail = Counter(), Counter()",
        "for item in raw:",
        "    seq = list(item.get('sequence') or [])",
        "    result = int(item.get('success_count', 0))",
        "    for task in seq[:result]:",
        "        cnt_success[task] += 1",
        "    if result < len(seq):",
        "        cnt_fail[seq[result]] += 1",
        "total = cnt_success + cnt_fail",
        "task_info = {task: {'success': int(cnt_success[task]), 'total': int(total[task])} for task in total}",
        "merged = {",
        "    'worker_count': worker_count,",
        "    'sequence_count': len(raw),",
        "    'avg_seq_len': sum(results) / len(results),",
        "    'chain_sr': chain_sr,",
        "    'task_info': task_info,",
        "    'workers': [str(path) for path in dirs],",
        "}",
        "base_dir.mkdir(parents=True, exist_ok=True)",
        "out_path = base_dir / 'merged_results.json'",
        "out_path.write_text(json.dumps(merged, indent=2, sort_keys=True))",
        "print('[starvla-calvin-merge] merged_results=' + str(out_path))",
        "print('[starvla-calvin-merge] sequence_count=' + str(merged['sequence_count']) + ' avg_seq_len=' + f\"{merged['avg_seq_len']:.4f}\")",
        "print('[starvla-calvin-merge] chain_sr=' + json.dumps(chain_sr, sort_keys=True))",
        "PY",
        "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_merge_completed; fi",
    ]
    return "\n".join(lines) + "\n"


def build_eval_commands(spec: EvalLaunch) -> tuple[dict[str, str], dict[str, Any]]:
    catalog = load_catalog()
    preset = dict(catalog["eval_presets"][spec.preset])
    port = spec.port or int(preset.get("default_port", 5694))
    commands: dict[str, str] = {}

    if spec.preset == "calvin":
        dataset_path = spec.dataset_path or preset.get("default_dataset_path")
        calvin_config_path = preset.get("default_calvin_config_path", "playground/Code/calvin/calvin_models/conf")
        eval_sequences_path = preset.get("default_eval_sequences_path", "examples/calvin/eval_files/eval_sequences.json")
        num_sequences = spec.trials or int(preset.get("default_num_sequences", 1000))
        max_steps_per_task = spec.max_steps_per_task or int(preset.get("default_max_steps_per_task", 360))
        calvin_render_backend = spec.calvin_render_backend or preset.get("default_render_backend", "auto")
        server_gpus = _resolve_eval_server_gpus(spec.server_gpu)
        if not server_gpus:
            server_gpus = [0]
        replicas_per_gpu = _eval_replicas_per_gpu(spec.eval_throughput_profile, spec.replicas_per_gpu, preset)
        requested_workers = spec.parallel_workers if spec.parallel_workers is not None else len(server_gpus) * replicas_per_gpu
        worker_count = max(1, int(requested_workers or 1))
        extra = " ".join(q(item) for item in spec.extra_args)

        ports: list[int] = []
        worker_gpu_assignment: list[int] = []
        for worker_i in range(worker_count):
            worker_port = port + worker_i
            worker_gpu = server_gpus[worker_i % len(server_gpus)]
            worker_gpu_assignment.append(worker_gpu)
            ports.append(worker_port)
            server_key = "server" if worker_count == 1 else f"server{worker_i}"
            client_key = "client" if worker_count == 1 else f"client{worker_i}"
            commands[server_key] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_prepare preset={q(spec.preset)} worker={q(worker_i)} port={q(worker_port)} gpu={q(worker_gpu)}; fi",
                    f"export your_ckpt={q(spec.ckpt)}",
                    f"export star_vla_python={q(DEFAULT_PYTHON)}",
                    f"export gpu_id={q(worker_gpu)}",
                    f"export port={q(worker_port)}",
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_launch; fi",
                    f"bash {q(preset['server_script'])}",
                ]
            ) + "\n"
            if not spec.start_client:
                continue
            if spec.eval_log_dir:
                worker_log_dir = str(Path(spec.eval_log_dir) / "workers" / f"{worker_i:03d}")
                eval_log_dir_export = f"export eval_log_dir={q(worker_log_dir)}"
            else:
                eval_log_dir_export = f'export eval_log_dir="${{STARVLA_JOB_DIR}}/calvin_eval/workers/{worker_i:03d}"'
            commands[client_key] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_prepare preset={q(spec.preset)} worker={q(worker_i)}; fi",
                    "export CUDA_VISIBLE_DEVICES=''",
                    f"export your_ckpt={q(spec.ckpt)}",
                    f"export calvin_python={q(DEFAULT_PYTHON)}",
                    f"export host={q('127.0.0.1')}",
                    f"export base_port={q(worker_port)}",
                    f"export dataset_path={q(dataset_path)}",
                    f"export calvin_config_path={q(calvin_config_path)}",
                    f"export eval_sequences_path={q(eval_sequences_path)}",
                    f"export num_sequences={q(num_sequences)}",
                    f"export max_steps_per_task={q(max_steps_per_task)}",
                    f"export sequence_start={q(worker_i)}",
                    f"export sequence_stride={q(worker_count)}",
                    f"export STARVLA_CALVIN_RENDER_BACKEND={q(calvin_render_backend)}",
                    f"export STARVLA_CALVIN_RENDER_GPU={q(worker_gpu)}",
                    eval_log_dir_export,
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_launch; fi",
                    f"bash {q(preset['client_script'])}" + (f" {extra}" if extra else ""),
                ]
            ) + "\n"
        if spec.start_client:
            commands["merge"] = _calvin_merge_command(worker_count, spec.eval_log_dir)

        meta = {
            "kind": "eval",
            "preset": spec.preset,
            "ckpt": spec.ckpt,
            "port": port,
            "ports": ports,
            "server_gpu": spec.server_gpu,
            "server_gpus_resolved": server_gpus,
            "worker_gpu_assignment": worker_gpu_assignment,
            "client_gpu": spec.client_gpu,
            "start_client": spec.start_client,
            "parallel_workers": worker_count,
            "replicas_per_gpu": replicas_per_gpu,
            "eval_throughput_profile": spec.eval_throughput_profile,
            "calvin_render_backend": calvin_render_backend,
            "trials": spec.trials,
            "num_sequences": num_sequences,
            "max_steps_per_task": max_steps_per_task,
            "dataset_path": dataset_path,
            "eval_log_dir": spec.eval_log_dir or "${STARVLA_JOB_DIR}/calvin_eval",
        }
        return commands, meta

    server = [
        f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_prepare preset={q(spec.preset)} port={q(port)}; fi",
        f"export CUDA_VISIBLE_DEVICES={q(spec.server_gpu)}",
        "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_launch; fi",
        f"{q(DEFAULT_PYTHON)} deployment/model_server/server_policy.py --ckpt_path {q(spec.ckpt)} --port {q(port)} --use_bf16",
    ]
    if preset.get("server_script") and spec.preset in {"calvin", "robotwin"}:
        if spec.preset == "calvin":
            server = [
                f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_prepare preset={q(spec.preset)} port={q(port)}; fi",
                f"export your_ckpt={q(spec.ckpt)}",
                f"export star_vla_python={q(DEFAULT_PYTHON)}",
                f"export gpu_id={q(spec.server_gpu)}",
                f"export port={q(port)}",
                "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_launch; fi",
                f"bash {q(preset['server_script'])}",
            ]
        elif spec.preset == "robotwin":
            server = [
                f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_prepare preset={q(spec.preset)} port={q(port)}; fi",
                "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_server_launch; fi",
                f"STARVLA_PYTHON={q(DEFAULT_PYTHON)} bash {q(preset['server_script'])} {q(spec.ckpt)} {q(spec.server_gpu)} {q(port)}",
            ]
    commands["server"] = "\n".join(server) + "\n"

    if spec.start_client:
        kind = preset.get("client_kind")
        if kind == "libero":
            task_suite = spec.task or preset.get("default_task_suite", "libero_goal")
            trials = spec.trials or int(preset.get("default_trials", 50))
            model_root = "$(dirname $(dirname " + q(spec.ckpt) + "))"
            commands["client"] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_prepare task={q(task_suite)}; fi",
                    "export MUJOCO_GL=${MUJOCO_GL:-egl}",
                    "export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}",
                    "export LIBERO_HOME=${LIBERO_HOME:-playground/Code/libero}",
                    "export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}",
                    "export LIBERO_Python=${LIBERO_Python:-python}",
                    "export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH:-}",
                    "export CUDA_VISIBLE_DEVICES=" + q(spec.client_gpu),
                    f"mkdir -p {model_root}/results/{q(task_suite)}",
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_launch; fi",
                    f"${{LIBERO_Python}} examples/LIBERO/eval_files/eval_libero.py --args.pretrained-path {q(spec.ckpt)} --args.host 127.0.0.1 --args.port {q(port)} --args.task-suite-name {q(task_suite)} --args.num-trials-per-task {q(trials)} --args.video-out-path {model_root}/results/{q(task_suite)}/interactive_eval",
                ]
            ) + "\n"
        elif spec.preset == "calvin":
            dataset_path = spec.dataset_path or preset.get("default_dataset_path")
            calvin_config_path = preset.get("default_calvin_config_path", "playground/Code/calvin/calvin_models/conf")
            eval_sequences_path = preset.get("default_eval_sequences_path", "examples/calvin/eval_files/eval_sequences.json")
            num_sequences = spec.trials or int(preset.get("default_num_sequences", 1000))
            max_steps_per_task = spec.max_steps_per_task or int(preset.get("default_max_steps_per_task", 360))
            calvin_render_backend = spec.calvin_render_backend or preset.get("default_render_backend", "auto")
            eval_log_dir_export = (
                f"export eval_log_dir={q(spec.eval_log_dir)}"
                if spec.eval_log_dir
                else 'export eval_log_dir="${STARVLA_JOB_DIR}/calvin_eval_logs"'
            )
            extra = " ".join(q(item) for item in spec.extra_args)
            commands["client"] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_prepare preset={q(spec.preset)}; fi",
                    f"export your_ckpt={q(spec.ckpt)}",
                    f"export calvin_python={q(DEFAULT_PYTHON)}",
                    f"export base_port={q(port)}",
                    f"export dataset_path={q(dataset_path)}",
                    f"export calvin_config_path={q(calvin_config_path)}",
                    f"export eval_sequences_path={q(eval_sequences_path)}",
                    f"export num_sequences={q(num_sequences)}",
                    f"export max_steps_per_task={q(max_steps_per_task)}",
                    f"export STARVLA_CALVIN_RENDER_BACKEND={q(calvin_render_backend)}",
                    f"export STARVLA_CALVIN_RENDER_GPU={q(spec.server_gpu)}",
                    eval_log_dir_export,
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_launch; fi",
                    f"bash {q(preset['client_script'])}" + (f" {extra}" if extra else ""),
                ]
            ) + "\n"
        elif spec.preset == "robotwin":
            task = spec.task or "adjust_bottle"
            commands["client"] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_prepare task={q(task)}; fi",
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_launch; fi",
                    f"bash {q(preset['client_script'])} {q(task)} {q(task)} starvla_eval 0 {q(spec.client_gpu)} {q(spec.ckpt)} {q(port)} 127.0.0.1",
                ]
            ) + "\n"
        elif preset.get("client_script"):
            commands["client"] = "\n".join(
                [
                    f"if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_prepare preset={q(spec.preset)}; fi",
                    "if declare -F _starvla_phase >/dev/null; then _starvla_phase eval_client_launch; fi",
                    f"bash {q(preset['client_script'])} client",
                ]
            ) + "\n"

    meta = {
        "kind": "eval",
        "preset": spec.preset,
        "ckpt": spec.ckpt,
        "port": port,
        "server_gpu": spec.server_gpu,
        "client_gpu": spec.client_gpu,
        "start_client": spec.start_client,
        "trials": spec.trials,
        "max_steps_per_task": spec.max_steps_per_task,
        "calvin_render_backend": spec.calvin_render_backend or preset.get("default_render_backend", "auto"),
        "dataset_path": spec.dataset_path or preset.get("default_dataset_path"),
        "eval_log_dir": spec.eval_log_dir or "${STARVLA_JOB_DIR}/calvin_eval_logs",
    }
    return commands, meta


def create_job_files(job_name: str, command: str, meta: dict[str, Any], window_name: str = "main") -> tuple[Path, Path, Path]:
    runs_root = ensure_runtime_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = runs_root / f"{timestamp}_{job_name}"
    job_dir.mkdir(parents=True, exist_ok=False)
    log_file, command_file, runner = _window_artifact_paths(job_dir, window_name)

    command_file.write_text(command, encoding="utf-8")
    header = _common_header(log_file, job_dir=job_dir)
    runner.write_text("\n".join(header) + "\n" + command + "\necho \"[starvla-interaction] exited at $(date -Is)\"\n", encoding="utf-8")
    runner.chmod(0o755)

    job_meta = dict(meta)
    roots = artifact_roots_report()
    job_meta.update(
        {
            "job_name": job_name,
            "job_dir": str(job_dir),
            "log_file": str(log_file),
            "combined_log": str(job_dir / "logs" / "all.log"),
            "logs_dir": str(job_dir / "logs"),
            "commands_dir": str(job_dir / "commands"),
            "runners_dir": str(job_dir / "runners"),
            "interaction_artifact_dir": str(job_dir),
            "desired_runs_root": str(DESIRED_RUNS_ROOT),
            "active_runs_root": str(runs_root),
            "fallback_runs_root": str(FALLBACK_RUNS_ROOT),
            "using_fallback_runs_root": roots["using_fallback_runs_root"],
        }
    )
    (job_dir / "job.json").write_text(json.dumps(job_meta, indent=2, sort_keys=True), encoding="utf-8")
    return job_dir, runner, log_file


def create_extra_window_files(job_dir: Path, window_name: str, command: str) -> tuple[Path, Path]:
    log_file, command_file, runner = _window_artifact_paths(job_dir, window_name)
    runner.write_text("\n".join(_common_header(log_file, job_dir=job_dir)) + "\n" + command + "\n", encoding="utf-8")
    runner.chmod(0o755)
    command_file.write_text(command, encoding="utf-8")
    return runner, log_file


def _window_artifact_paths(job_dir: Path, window_name: str) -> tuple[Path, Path, Path]:
    logs_dir = job_dir / "logs"
    commands_dir = job_dir / "commands"
    runners_dir = job_dir / "runners"
    logs_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)
    runners_dir.mkdir(parents=True, exist_ok=True)
    return (
        logs_dir / f"{window_name}.log",
        commands_dir / f"{window_name}.command.sh",
        runners_dir / f"{window_name}.runner.sh",
    )
