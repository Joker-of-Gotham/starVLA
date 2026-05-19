#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interaction.core.catalog import (
    all_action_expert_keys,
    all_model_keys,
    base_model_key,
    dataset_name,
    default_model_for_framework,
    experiment_group_name,
    keys,
    load_catalog,
    load_policy_catalog,
    policy_combo_slug,
    policy_keys,
    resolve_action_expert,
    resolve_repo_path,
)
from interaction.core.checks import dataset_quick_report, environment_report, model_quick_report
from interaction.core.commands import EvalLaunch, TrainLaunch, build_eval_commands, build_train_command, create_extra_window_files, create_job_files
from interaction.core.diagnostics import compact_value
from interaction.core.gpu import command_exists, detect_gpus, format_gpu_table, gpu_nvlink_status, gpu_topology, run_text_command, select_gpus
from interaction.core.monitor import collect_job_snapshot, find_job, iter_jobs, parse_latest_metrics, render_snapshot, snapshot_to_text, tail, watch
from interaction.core.perf import perf_preflight, resolve_perf_settings
from interaction.core.paths import DEFAULT_CHECKPOINT_ROOT, REPO_ROOT, artifact_roots_report
from interaction.core.tmux import attach as tmux_attach
from interaction.core.tmux import kill_session, new_session, new_window, session_exists, tmux_available
from interaction.core.ui import console as rich_console
from interaction.core.ui import cprint, print_table, remove_markup, status_markup


def _interactive_confirm(title: str, rows: list[list[Any]], command_preview: str | None = None, yes: bool = False) -> bool:
    if yes or not sys.stdin.isatty():
        return True
    print_table(title, ["item", "value"], rows)
    if command_preview:
        cprint("[bold]Command preview[/bold]")
        print(command_preview.strip())
    raw = input("Confirm launch? Type 'yes' or 'y' to start [default: no]: ").strip().lower()
    return raw in {"y", "yes"}


def parse_num_gpus(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "all", "auto", "none"}:
        return None
    try:
        count = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer, 'all', or 'auto'") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("--num-gpus must be positive, 'all', or 'auto'")
    return count


def _ask_num_gpus(prompt: str = "Number of GPUs when auto [all]: ") -> int | None:
    while True:
        try:
            return parse_num_gpus(input(prompt).strip())
        except argparse.ArgumentTypeError as exc:
            cprint(f"[red]Invalid GPU count:[/red] {exc}")


def split_multi_values(values: list[Any] | None) -> list[str]:
    out: list[str] = []
    for item in values or []:
        if isinstance(item, (list, tuple)):
            out.extend(split_multi_values(list(item)))
            continue
        for part in str(item).replace(",", " ").split():
            text = part.strip()
            if text and text.lower() not in {"none", "auto", "default"} and text not in out:
                out.append(text)
    return out


def normalize_run_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "auto"} else text


def _model_runtime_status(model_path: str | None) -> tuple[bool, str]:
    if not model_path:
        return False, "missing model path"
    resolved = resolve_repo_path(model_path)
    model_text = str(resolved or model_path).lower()
    if "qwen3.5" not in model_text and "qwen3_5" not in model_text:
        return True, "no special runtime class required"
    code = r"""
source bar/activate_starvla_no_proxy.sh
python - <<'PY'
import transformers
print("OK" if hasattr(transformers, "Qwen3_5ForConditionalGeneration") else "MISSING")
PY
"""
    proc = subprocess.run(["bash", "-lc", code], cwd=REPO_ROOT, text=True, capture_output=True, timeout=60)
    ok = proc.returncode == 0 and proc.stdout.strip().splitlines()[-1:] == ["OK"]
    if ok:
        return True, "Qwen3_5ForConditionalGeneration available"
    detail = (proc.stdout + proc.stderr).strip()[-500:] or "Qwen3_5ForConditionalGeneration missing"
    return False, detail


def _confirm_train_launch(meta: dict[str, Any], command: str, yes: bool = False) -> bool:
    rows = [
        ["preset", meta.get("preset")],
        ["mode", meta.get("mode")],
        ["dataset_key", meta.get("dataset_key") or meta.get("dataset") or meta.get("data_mix")],
        ["training_policies", ",".join(meta.get("training_policies") or []) or "-"],
        ["base_model", meta.get("base_model_key") or meta.get("model_key") or meta.get("model")],
        ["action_expert", meta.get("action_expert") or meta.get("framework")],
        ["action_expert_resolved", meta.get("action_expert_resolved") or meta.get("framework")],
        ["structure_policies", ",".join(meta.get("structure_policies") or []) or "-"],
        ["dataset_dims", f"a={meta.get('dataset_action_dim') or '-'} s={meta.get('dataset_state_dim') or '-'} h={meta.get('dataset_horizon') or '-'}"],
        ["model_path", meta.get("model")],
        ["experiment_group", meta.get("experiment_group")],
        ["run_id", meta.get("run_id")],
        ["run_type", meta.get("start_mode") or ("continue/resume" if meta.get("resume") else "new")],
        ["init_checkpoint", meta.get("init_checkpoint") or "-"],
        ["reload_modules", meta.get("reload_modules") or "-"],
        ["gpus", ",".join(map(str, meta.get("gpus") or [])) or "CPU/auto-unset"],
        ["num_processes", meta.get("num_processes")],
        ["mixed_precision", meta.get("mixed_precision")],
        ["freeze_modules", meta.get("freeze_modules") or "(none)"],
        ["perf_profile", meta.get("perf_profile")],
        ["throughput_profile", meta.get("throughput_profile")],
        ["vla_batch_size", meta.get("vla_batch_size")],
        ["vlm_batch_size", meta.get("vlm_batch_size")],
        ["dataloader_workers", meta.get("dataloader_workers")],
        ["dataloader_prefetch_factor", meta.get("dataloader_prefetch_factor")],
        ["accelerate_config", meta.get("accelerate_config")],
        ["max_train_steps", meta.get("max_train_steps")],
        ["keep_top_k", meta.get("keep_top_k")],
        ["output_dir", meta.get("output_dir")],
        ["checkpoint_dir", meta.get("checkpoint_dir")],
        ["full_state_dir", meta.get("full_state_dir")],
    ]
    launch_line = next((line for line in reversed(command.splitlines()) if line.strip().startswith("accelerate ")), command)
    return _interactive_confirm("Confirm Training Launch", rows, launch_line, yes=yes)


def _confirm_eval_launch(meta: dict[str, Any], commands: dict[str, str], yes: bool = False) -> bool:
    rows = [
        ["preset", meta.get("preset")],
        ["checkpoint", meta.get("ckpt")],
        ["port", meta.get("port")],
        ["ports", meta.get("ports")],
        ["server_gpu", meta.get("server_gpu")],
        ["server_gpus_resolved", meta.get("server_gpus_resolved")],
        ["worker_gpu_assignment", meta.get("worker_gpu_assignment")],
        ["client_gpu", meta.get("client_gpu")],
        ["start_client", meta.get("start_client")],
        ["eval_throughput_profile", meta.get("eval_throughput_profile")],
        ["parallel_workers", meta.get("parallel_workers")],
        ["replicas_per_gpu", meta.get("replicas_per_gpu")],
        ["num_sequences", meta.get("num_sequences") or meta.get("trials")],
        ["max_steps_per_task", meta.get("max_steps_per_task")],
        ["session", meta.get("tmux_session")],
    ]
    command_preview = _compact_command_preview(commands)
    return _interactive_confirm("Confirm Evaluation Launch", rows, command_preview, yes=yes)


def _compact_command_preview(commands: dict[str, str], max_commands: int = 6, max_chars: int = 5000) -> str:
    items = list(commands.items())
    if len(items) > max_commands:
        head_count = max_commands - 2
        shown = items[:head_count] + items[-2:]
        omitted = len(items) - len(shown)
    else:
        shown = items
        omitted = 0
    parts: list[str] = []
    for name, command in shown:
        text = command.strip()
        if len(text) > 900:
            text = text[:900].rstrip() + "\n... <truncated>"
        parts.append(f"# {name}\n{text}")
    if omitted:
        parts.insert(head_count, f"# ...\n{omitted} worker command(s) omitted from preview. Full scripts are written under commands/ in the job dir.")
    preview = "\n\n".join(parts)
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "\n... <preview truncated>"
    return preview


def sanitize_session(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not value:
        value = "job"
    return value[:80]


def default_run_id(prefix: str) -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%m%d_%H%M%S')}"


def latest_checkpoint_run_id(prefix: str, root: str | Path | None = None) -> str | None:
    root = Path(root) if root is not None else DEFAULT_CHECKPOINT_ROOT
    if not root.exists():
        return None
    all_candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and ((path / "accelerate_state").exists() or (path / "checkpoints").exists())
    ]
    candidates = [path for path in all_candidates if path.name.startswith(prefix)] or all_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def can_write_dir(path: str | Path) -> bool:
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".starvla_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _artifact_status(key: str, value: Any) -> Any:
    if key == "using_fallback_runs_root":
        if isinstance(value, bool):
            return status_markup(True, ok_text="NO") if not value else f"{status_markup('warn')} YES"
        return value
    if key.endswith("_writable") and value is False:
        return f"{status_markup('warn')} NO"
    if isinstance(value, bool):
        return status_markup(value)
    return value


def _path_state(value: str | Path | None) -> tuple[str, bool]:
    if value in {None, ""}:
        return "-", False
    path = resolve_repo_path(value)
    if path is None:
        return str(value), False
    return str(path), path.exists()


def _train_preflight_rows(meta: dict[str, Any], allow_readonly_output: bool = False) -> tuple[list[list[Any]], bool]:
    checks = [
        ("trainer_script", meta.get("trainer_script"), True),
        ("config_yaml", meta.get("config_yaml"), True),
        ("data_root", meta.get("data_root"), bool(meta.get("data_root"))),
        ("base_model", meta.get("model"), True),
        ("accelerate_config", "starVLA/config/deepseeds/deepspeed_zero2.yaml", True),
    ]
    if meta.get("init_checkpoint"):
        checks.append(("init_checkpoint", meta.get("init_checkpoint"), True))
    rows: list[list[Any]] = []
    failed = False
    for name, value, required in checks:
        path, exists = _path_state(value)
        ok = exists or not required
        failed = failed or not ok
        rows.append([name, status_markup(ok), path])
    output_root = meta.get("run_root_dir")
    output_ok = can_write_dir(output_root) if output_root else False
    failed = failed or (not output_ok and not allow_readonly_output)
    output_status = status_markup(output_ok) if output_ok or not allow_readonly_output else status_markup("warn") + " allowed"
    rows.append(["output_root_writable", output_status, output_root or "-"])
    model_ok, model_detail = _model_runtime_status(meta.get("model"))
    failed = failed or not model_ok
    rows.append(["model_runtime", status_markup(model_ok), model_detail])
    accel_path, accel_exists = _path_state(meta.get("accelerate_config"))
    failed = failed or not accel_exists
    rows.append(["accelerate_config", status_markup(accel_exists), accel_path])
    for warning in meta.get("perf_warnings") or []:
        rows.append(["perf_warning", status_markup("warn"), warning])
    for warning in meta.get("throughput_warnings") or []:
        rows.append(["throughput_warning", status_markup("warn"), warning])
    for warning in (meta.get("policy_launch") or {}).get("warnings") or []:
        rows.append(["policy_warning", status_markup("warn"), warning])
    for note in meta.get("throughput_notes") or []:
        rows.append(["throughput_note", status_markup("ok"), note])
    return rows, failed


def _print_launch_failure(stage: str, exc: BaseException) -> None:
    cprint(f"[red]Failed during {stage}:[/red] {exc}")
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.stdout:
            cprint(f"[yellow]stdout:[/yellow]\n{exc.stdout}")
        if exc.stderr:
            cprint(f"[yellow]stderr:[/yellow]\n{exc.stderr}")
    cprint("[cyan]Root locator:[/cyan] the tmux session was not started; fix the message above before retrying.")


def _metric_label(key: str) -> str:
    key_lower = key.lower()
    if key_lower == "mse_score":
        return "mse"
    if key_lower.endswith("loss") or key_lower == "loss":
        return key.replace("/", ".")
    return key.replace("/", ".")


def command_check(args: argparse.Namespace) -> int:
    report = environment_report(include_action_heads=not args.skip_action_heads)
    rows = [
        ["repo", report["repo_root"]],
        ["activate", status_markup(report["activate_script"])],
        ["python", f"{status_markup(report['python_exists'])} {report['python']}"],
        ["tmux", status_markup(report["tmux"])],
        ["proxy_clean", status_markup(report["proxy_clean"])],
        ["cuda_available", status_markup(bool(report.get("cuda_available")), ok_text="YES", fail_text="NO")],
        ["torch_device_count", report.get("cuda_device_count", "-")],
        ["torch_cuda", report.get("torch_cuda_version") or "-"],
        ["CUDA_VISIBLE_DEVICES", report.get("cuda_visible_devices") or "unset"],
    ]
    print_table("Environment", ["item", "value"], rows)

    gpu_rows = [
        [
            row["id"],
            row["name"],
            status_markup(row["h200"] == "yes", ok_text="H200", fail_text="other"),
            row["used"],
            row["free"],
            row["util"],
            row["temp"],
            row["power"],
            row["power_pct"],
            row["sm_clock"],
            row["mem_clock"],
            row["pstate"],
        ]
        for row in format_gpu_table(detect_gpus())
    ]
    print_table("GPUs", ["id", "name", "H200", "used", "free", "util", "temp", "power", "power%", "sm", "mem", "pstate"], gpu_rows or [["-", "no GPU detected", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]])

    artifacts = report.get("artifacts", {})
    artifact_rows = [[key, _artifact_status(key, value)] for key, value in artifacts.items()]
    print_table("Artifact roots", ["item", "value"], artifact_rows)

    import_rows = [[name, status_markup("failed" if str(value).startswith("ERROR") else "warn" if str(value).startswith("OPTIONAL") else "ok") + f" {value}"] for name, value in sorted(report.get("imports", {}).items())]
    print_table("Python imports", ["module", "status"], import_rows)

    runtime_rows = [
        [name, status_markup("failed" if str(value).startswith("ERROR") else "ok") + f" {value}"]
        for name, value in sorted(report.get("runtime_capabilities", {}).items())
    ]
    print_table("Runtime capabilities", ["capability", "status"], runtime_rows)

    if report.get("import_check_parse_error"):
        print_table(
            "Import Check Diagnostics",
            ["item", "value"],
            [
                ["returncode", report.get("import_check_rc")],
                ["parse_error", report.get("import_check_parse_error")],
                ["tail", report.get("import_check_output", "").strip()[-2000:]],
            ],
        )

    path_rows = [[item["path"], status_markup(item["exists"])] for item in report.get("required_paths", [])]
    print_table("Required paths", ["path", "exists"], path_rows)

    model_report = model_quick_report()
    model_rows = [[key, data["label"], data["path"], status_markup(data["exists"])] for key, data in model_report.items()]
    print_table("Model presets", ["model", "label", "path", "exists"], model_rows)

    ds_report = dataset_quick_report(include_optional_missing=True)
    ds_rows = [[key, data["data_mix"], data["data_root"], status_markup(data["exists"])] for key, data in ds_report.items()]
    print_table("Dataset presets", ["preset", "mix", "root", "exists"], ds_rows)

    if report.get("action_heads") is not None:
        action = report["action_heads"]
        print_table("Action heads", ["status", "tail"], [[status_markup(action["returncode"] == 0), action["output"].strip()[-1000:]]])

    failed_paths = [item for item in report.get("required_paths", []) if not item["exists"]]
    failed_imports = [name for name, value in report.get("imports", {}).items() if str(value).startswith("ERROR")]
    failed_runtime = [name for name, value in report.get("runtime_capabilities", {}).items() if str(value).startswith("ERROR")]
    failed_models = [key for key, data in model_report.items() if not data["exists"]]
    failed_datasets = [key for key, data in ds_report.items() if not data["exists"] and not data.get("optional")]
    warning_items = []
    if artifacts.get("checkpoint_root_writable") is False:
        warning_items.append("default checkpoint_root is not writable; pass --run-root-dir to a writable path before training")
    if artifacts.get("desired_runs_root_writable") is False:
        warning_items.append("desired interaction runs_root is not writable; metadata will use the fallback runs root")
    missing_optional_dataset_presets = [key for key, data in ds_report.items() if not data["exists"] and data.get("optional")]
    if missing_optional_dataset_presets:
        warning_items.append("missing optional dataset preset roots: " + ", ".join(missing_optional_dataset_presets))
    if failed_paths or failed_imports or failed_runtime or failed_models or failed_datasets or report.get("proxy_clean") is False:
        cprint("[red]Environment check failed.[/red] Fix the red rows first; they are direct launch blockers.")
        if failed_runtime:
            cprint("[red]missing runtime capabilities:[/red] " + ", ".join(failed_runtime))
        if failed_models:
            cprint("[red]missing model presets:[/red] " + ", ".join(failed_models))
        if failed_datasets:
            cprint("[red]missing dataset presets:[/red] " + ", ".join(failed_datasets))
        return 1
    if report.get("action_heads") and report["action_heads"]["returncode"] != 0:
        cprint("[red]Action-head check failed.[/red] The tail above is the immediate failure context.")
        return 1
    if warning_items:
        cprint("[yellow]Environment check passed with warnings.[/yellow]")
        for item in warning_items:
            cprint(f"[yellow]- {item}[/yellow]")
        return 0
    cprint("[green]Environment check passed.[/green]")
    return 0


def command_perf_check(args: argparse.Namespace) -> int:
    gpus = select_gpus(args.gpus, args.num_gpus, args.min_free_mb)
    settings = resolve_perf_settings(
        profile=args.perf_profile,
        selected_gpus=gpus,
        nccl_debug=args.nccl_debug,
        nccl_trace=args.nccl_trace,
        nccl_socket_ifname=args.nccl_socket_ifname,
        nccl_ib_hca=args.nccl_ib_hca,
        nccl_p2p_level=None if args.nccl_p2p_level == "auto" else args.nccl_p2p_level,
        env_overrides=args.nccl_env or [],
    )
    gpu_rows = []
    selected_set = set(gpus)
    for row in format_gpu_table(detect_gpus()):
        gpu_rows.append(
            [
                "*" if int(row["id"]) in selected_set else "",
                row["id"],
                row["name"],
                status_markup(row["h200"] == "yes", ok_text="H200", fail_text="other"),
                row["used"],
                row["free"],
                row["util"],
                row["power"],
                row["power_pct"],
                row["sm_clock"],
                row["mem_clock"],
                row["pstate"],
            ]
        )
    print_table("GPU Performance Inventory", ["sel", "id", "name", "type", "used", "free", "util", "power", "power%", "sm", "mem", "pstate"], gpu_rows or [["-", "-", "no GPU detected", "-", "-", "-", "-", "-", "-", "-", "-", "-"]])
    perf_rows = [[name, status_markup(status) if isinstance(status, bool) else status_markup(status) if isinstance(status, str) else status, detail] for name, status, detail in perf_preflight(gpus, settings)]
    print_table("Performance Profile", ["check", "status", "detail"], perf_rows)
    env_rows = [[key, value, "forced" if key in settings.env_forced else "default-if-unset"] for key, value in sorted((settings.env_defaults | settings.env_forced).items())]
    print_table("NCCL / Torch Env", ["env", "value", "mode"], env_rows or [["-", "-", "profile disabled"]])

    topo = gpu_topology()
    print_table("Topology", ["item", "value"], [["returncode", topo.get("returncode")], ["nvidia-smi topo -m tail", str(topo.get("output", "")).strip()[-4000:] or "-"]])
    nvlink = gpu_nvlink_status()
    if nvlink.get("returncode") == 0:
        print_table("NVLink", ["item", "value"], [["nvidia-smi nvlink -s tail", str(nvlink.get("output", "")).strip()[-4000:] or "-"]])

    cprint("[cyan]Bandwidth validation commands:[/cyan]")
    cprint(f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus)) or '<ids>'} all_reduce_perf -b 8 -e {args.nccl_test_max_bytes} -f 2 -g {max(len(gpus), 1)}")
    cprint(f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus)) or '<ids>'} p2pBandwidthLatencyTest")
    cprint(f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus)) or '<ids>'} nvbandwidth")
    if args.run_nccl_test:
        if not command_exists("all_reduce_perf"):
            cprint("[red]all_reduce_perf was not found in PATH; build NVIDIA nccl-tests first.[/red]")
            return 4
        env_prefix = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))}" if gpus else ""
        cmd = ["bash", "-lc", f"{env_prefix} all_reduce_perf -b 8 -e {args.nccl_test_max_bytes} -f 2 -g {max(len(gpus), 1)}"]
        rc, out = run_text_command(cmd, timeout=args.timeout)
        print(out[-12000:])
        return rc
    return 0


def command_catalog(args: argparse.Namespace) -> int:
    catalog = load_catalog()
    policies = load_policy_catalog()
    kind = args.kind

    def show_datasets() -> None:
        rows = []
        for key, item in policies.get("datasets", {}).items():
            rows.append(
                [
                    key,
                    item.get("label", key),
                    item.get("preset", "-"),
                    item.get("data_mix") or item.get("vlm_dataset") or "-",
                    item.get("embodiment", "-"),
                    f"{item.get('action_dim', '-')}/{item.get('state_dim', '-')}/{item.get('horizon', '-')}",
                    ",".join(item.get("recommended_training_policies", [])[:5]),
                    ",".join(item.get("recommended_structure_policies", [])[:5]),
                ]
            )
        print_table("Datasets", ["id", "label", "preset", "mix", "embodiment", "a/s/h", "train policy", "structure"], rows)

    def show_models() -> None:
        rows = []
        for key in all_model_keys():
            entry = policies.get("base_models", {}).get(key, {})
            model_key = entry.get("model", key)
            model_entry = catalog.get("models", {}).get(model_key, {})
            path = model_entry.get("path", model_key)
            resolved = resolve_repo_path(path)
            rows.append([key, entry.get("family", "-"), entry.get("label") or model_entry.get("label", key), model_key, path, status_markup(bool(resolved and resolved.exists()))])
        print_table("Base Models", ["id", "family", "label", "catalog_key", "path", "exists"], rows)

    def show_experts() -> None:
        rows = []
        for key in all_action_expert_keys():
            entry = policies.get("action_experts", {}).get(key, {})
            fw = entry.get("default_framework", key)
            family_routes = entry.get("framework_by_family", {})
            rows.append([key, entry.get("label") or catalog.get("frameworks", {}).get(key, {}).get("label", key), fw, json.dumps(family_routes, sort_keys=True) if family_routes else "-", ",".join(entry.get("training_policies", [])), ",".join(entry.get("structure_policies", []))])
        print_table("Action Experts", ["id", "label", "default/framework", "family_routes", "T", "S"], rows)

    def show_training() -> None:
        rows = [[key, item.get("name", key), item.get("stage", "-"), item.get("status", "-"), ",".join(item.get("preferred_experts", item.get("preferred_modes", [])))] for key, item in policies.get("training_policies", {}).items()]
        print_table("Training Policies", ["id", "name", "stage", "status", "preferred"], rows)

    def show_structure() -> None:
        rows = [[key, item.get("name", key), item.get("target", "-"), item.get("status", "-"), ",".join(item.get("preferred_experts", []))] for key, item in policies.get("structure_policies", {}).items()]
        print_table("Structure Policies", ["id", "name", "target", "status", "preferred"], rows)

    if kind in {"all", "datasets"}:
        show_datasets()
    if kind in {"all", "base-models", "models"}:
        show_models()
    if kind in {"all", "action-experts", "experts"}:
        show_experts()
    if kind in {"all", "training-policies", "training"}:
        show_training()
    if kind in {"all", "structure-policies", "structure"}:
        show_structure()
    return 0


def _resolve_train_defaults(args: argparse.Namespace) -> TrainLaunch:
    catalog = load_catalog()
    policy_catalog = load_policy_catalog()
    dataset_key = getattr(args, "dataset", None)
    dataset_entry = policy_catalog.get("datasets", {}).get(dataset_key or "", {})
    training_policies = split_multi_values(getattr(args, "training_policy", []))
    structure_policies = split_multi_values(getattr(args, "structure_policy", []))
    preset_key = dataset_entry.get("preset", args.preset) if dataset_key else args.preset
    if "T12" in training_policies:
        cotrain_candidates = []
        if preset_key.endswith("_vla"):
            cotrain_candidates.append(preset_key[:-4] + "_cotrain")
        cotrain_candidates.append(preset_key + "_cotrain")
        for candidate in cotrain_candidates:
            if candidate in catalog["train_presets"]:
                preset_key = candidate
                break
    if preset_key not in catalog["train_presets"]:
        raise ValueError(f"Dataset {dataset_key} points to unknown train preset {preset_key}")
    preset = catalog["train_presets"][preset_key]
    requested_model = getattr(args, "base_model", None) or args.model or preset.get("model")
    requested_action_expert = getattr(args, "action_expert", None) or args.framework or preset.get("framework", "QwenPI")
    model = base_model_key(requested_model or default_model_for_framework(requested_action_expert))
    framework, _action_entry = resolve_action_expert(requested_action_expert, model)
    if not model:
        model = default_model_for_framework(framework)
    prefix = preset.get("run_id_prefix", args.preset)
    data_root = args.data_root if args.data_root is not None else dataset_entry.get("data_root")
    data_mix = args.data_mix if args.data_mix is not None else dataset_entry.get("data_mix", preset.get("data_mix"))
    vlm_dataset = args.vlm_dataset if args.vlm_dataset is not None else dataset_entry.get("vlm_dataset", preset.get("vlm_dataset"))
    dataset = dataset_key or dataset_name(preset_key, data_mix=data_mix, vlm_dataset=vlm_dataset)
    combo_slug = policy_combo_slug(training_policies, structure_policies)
    base_group = experiment_group_name(model, framework, dataset)
    if combo_slug:
        base_group = f"{base_group}-{combo_slug}"
    experiment_group = args.experiment_group or base_group
    checkpoint_root = args.run_root_dir or preset.get("run_root_dir", str(DEFAULT_CHECKPOINT_ROOT))
    group_root = Path(checkpoint_root) / experiment_group
    run_mode = "continue" if args.resume else args.run_mode
    run_id = normalize_run_id(args.run_id)
    if not run_id and run_mode == "continue" and not args.resume_from_checkpoint:
        run_id = latest_checkpoint_run_id(prefix, group_root)
        if not run_id:
            raise ValueError(f"No existing run found under {group_root}; pass --run-id or use --run-mode new.")
    run_id = run_id or default_run_id(prefix)
    init_checkpoint = normalize_run_id(getattr(args, "init_checkpoint", None))
    reload_modules = normalize_run_id(getattr(args, "reload_modules", None))
    if init_checkpoint and (run_mode == "continue" or args.resume_from_checkpoint):
        raise ValueError("--init-checkpoint/--posttrain-from starts a new post-training run; use resume only to continue an existing run.")
    gpus = select_gpus(args.gpus, args.num_gpus, args.min_free_mb)
    extra_args = args.extra or []
    return TrainLaunch(
        preset=preset_key,
        framework=framework,
        model=model,
        gpus=gpus,
        run_id=run_id,
        dataset_key=dataset_key,
        base_model_key=requested_model,
        action_expert_key=requested_action_expert,
        training_policies=training_policies,
        structure_policies=structure_policies,
        max_train_steps=args.max_steps,
        vla_batch_size=args.vla_batch_size,
        vlm_batch_size=args.vlm_batch_size,
        data_root=data_root,
        data_mix=data_mix,
        vlm_dataset=vlm_dataset,
        run_root_dir=args.run_root_dir,
        experiment_group=experiment_group,
        resume=(run_mode == "continue") or bool(args.resume_from_checkpoint),
        resume_from_checkpoint=args.resume_from_checkpoint,
        resume_mode=args.resume_mode,
        init_checkpoint=init_checkpoint,
        reload_modules=reload_modules,
        save_full_state=not args.no_full_state,
        keep_top_k=args.keep_top_k,
        freeze_modules=args.freeze_modules,
        mixed_precision=args.mixed_precision,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        main_process_ip=args.main_process_ip,
        main_process_port=args.main_process_port,
        disable_wandb=not args.enable_wandb,
        perf_profile=args.perf_profile,
        nccl_debug=args.nccl_debug,
        nccl_trace=args.nccl_trace,
        nccl_socket_ifname=args.nccl_socket_ifname,
        nccl_ib_hca=args.nccl_ib_hca,
        nccl_p2p_level=None if args.nccl_p2p_level == "auto" else args.nccl_p2p_level,
        nccl_env=args.nccl_env or [],
        throughput_profile=args.throughput_profile,
        dataloader_workers=args.dataloader_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        extra_args=extra_args,
    )


def command_train(args: argparse.Namespace) -> int:
    try:
        spec = _resolve_train_defaults(args)
    except ValueError as exc:
        cprint(f"[red]{exc}[/red]")
        return 2
    try:
        command, meta = build_train_command(spec)
    except ValueError as exc:
        cprint(f"[red]{exc}[/red]")
        return 2
    session = sanitize_session(args.session or f"starvla_{spec.run_id}")
    meta["tmux_session"] = session

    if args.dry_run:
        print(command)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    if not spec.gpus and not getattr(args, "allow_cpu", False):
        cprint("[red]No GPU was selected for training.[/red]")
        cprint("Use --gpus auto/all/<ids> after checking nvidia-smi visibility, or pass --allow-cpu only for a deliberate CPU smoke run.")
        return 4
    detected_gpu_ids = {gpu.index for gpu in detect_gpus()}
    missing_gpu_ids = [gpu for gpu in spec.gpus if detected_gpu_ids and gpu not in detected_gpu_ids]
    if missing_gpu_ids:
        cprint(f"[red]Selected GPU id(s) are not visible to nvidia-smi:[/red] {missing_gpu_ids}")
        cprint("Run perf-check or adjust --gpus/CUDA_VISIBLE_DEVICES before launching.")
        return 4
    if not can_write_dir(meta["run_root_dir"]) and not args.allow_readonly_output:
        cprint(f"[red]Training output root is not writable:[/red] {meta['run_root_dir']}")
        cprint("Grant write permission to that directory or pass --run-root-dir to a writable target. Use --allow-readonly-output only if the tmux runtime will have different permissions.")
        return 3
    preflight_rows, preflight_failed = _train_preflight_rows(meta, allow_readonly_output=args.allow_readonly_output)
    print_table("Training Preflight", ["check", "status", "resolved path"], preflight_rows)
    if preflight_failed:
        cprint("[red]Training preflight failed.[/red] The red row is the direct blocker; fix it before launching.")
        return 4
    perf_settings = resolve_perf_settings(
        profile=args.perf_profile,
        selected_gpus=spec.gpus,
        nccl_debug=args.nccl_debug,
        nccl_trace=args.nccl_trace,
        nccl_socket_ifname=args.nccl_socket_ifname,
        nccl_ib_hca=args.nccl_ib_hca,
        nccl_p2p_level=None if args.nccl_p2p_level == "auto" else args.nccl_p2p_level,
        env_overrides=args.nccl_env or [],
    )
    raw_perf_rows = perf_preflight(spec.gpus, perf_settings)
    perf_rows = [[name, status_markup(status) if isinstance(status, bool) else status, detail] for name, status, detail in raw_perf_rows]
    print_table("Performance Preflight", ["check", "status", "detail"], perf_rows)
    hard_perf_blockers = {
        "free_memory_check",
        "compute_process_check",
    }
    failed_perf = [row for row in raw_perf_rows if row[0] in hard_perf_blockers and row[1] is False]
    if failed_perf:
        cprint("[red]Performance preflight failed.[/red] Selected GPU(s) are already occupied or do not have enough free memory.")
        cprint("Stop the existing job, choose different GPUs, or use `bash interaction/bin/starvla-force-stop.sh <session>` for managed StarVLA sessions.")
        return 4
    if not tmux_available():
        cprint("[red]tmux is required for managed launches but is not available.[/red]")
        return 2
    if not _confirm_train_launch(meta, command, yes=getattr(args, "yes", False)):
        cprint("Launch cancelled.")
        return 130
    job_dir, runner, log_file = create_job_files(spec.run_id, command, meta)
    meta_path = job_dir / "job.json"
    current = json.loads(meta_path.read_text(encoding="utf-8"))
    current["tmux_session"] = session
    current["log_file"] = str(log_file)
    meta_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")

    try:
        new_session(session, runner, REPO_ROOT, "train")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        _print_launch_failure("tmux session creation", exc)
        return 2
    cprint(f"[green]Started training tmux session:[/green] {session}")
    cprint(f"Job dir: {job_dir}")
    cprint(f"Log: {log_file}")
    cprint(f"[cyan]Live monitor:[/cyan] bash interaction/bin/starvla-interact.sh monitor {job_dir.name}")
    if args.attach:
        tmux_attach(session)
    return 0


def command_eval(args: argparse.Namespace) -> int:
    if not args.ckpt:
        cprint("[red]--ckpt is required for eval launch.[/red]")
        return 2
    ckpt_path = resolve_repo_path(args.ckpt)
    if ckpt_path is not None and not ckpt_path.exists():
        cprint(f"[red]Checkpoint path does not exist:[/red] {ckpt_path}")
        cprint("This is a direct eval blocker; pass a valid checkpoint directory or weight file.")
        return 4
    if args.preset == "calvin" and args.parallel_workers and args.parallel_workers > 64 and not getattr(args, "allow_many_workers", False):
        cprint(
            "[red]Refusing to launch more than 64 CALVIN eval workers by default.[/red]\n"
            "Each worker creates one tmux window pair, one policy model copy, and one simulator. "
            "Use --allow-many-workers only after confirming memory, CPU, and tmux limits."
        )
        return 2
    spec = EvalLaunch(
        preset=args.preset,
        ckpt=args.ckpt,
        server_gpu=args.server_gpu,
        client_gpu=args.client_gpu,
        port=args.port,
        start_client=not args.server_only,
        task=args.task,
        trials=args.trials,
        max_steps_per_task=args.max_steps_per_task,
        parallel_workers=args.parallel_workers,
        replicas_per_gpu=args.replicas_per_gpu,
        eval_throughput_profile=args.eval_throughput_profile,
        calvin_render_backend=args.calvin_render_backend,
        dataset_path=args.dataset_path,
        eval_log_dir=args.eval_log_dir,
        extra_args=args.extra or [],
    )
    commands, meta = build_eval_commands(spec)
    session = sanitize_session(args.session or f"starvla_eval_{Path(args.ckpt).stem}")
    meta["tmux_session"] = session

    if args.dry_run:
        for name, command in commands.items():
            print(f"### {name}\n{command}")
        return 0
    if not tmux_available():
        cprint("[red]tmux is required for managed launches but is not available.[/red]")
        return 2
    if not _confirm_eval_launch(meta, commands, yes=getattr(args, "yes", False)):
        cprint("Launch cancelled.")
        return 130

    ordered_names = list(commands)
    primary_name = "server" if "server" in commands else ordered_names[0]
    job_dir, primary_runner, primary_log = create_job_files(session, commands[primary_name], meta, primary_name)
    try:
        new_session(session, primary_runner, REPO_ROOT, primary_name)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        _print_launch_failure("eval tmux session creation", exc)
        return 2
    window_logs: dict[str, str] = {primary_name: str(primary_log)}
    for name, command in commands.items():
        if name == primary_name:
            continue
        window_command = command
        launch_delay = _window_launch_delay(name, args.client_delay, getattr(args, "launch_stagger_seconds", 0.5))
        if launch_delay > 0:
            window_command = f"sleep {launch_delay:.2f}\n" + window_command
        window_runner, window_log = create_extra_window_files(job_dir, name, window_command)
        try:
            new_window(session, name, window_runner, REPO_ROOT)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            _print_launch_failure(f"eval tmux window creation ({name})", exc)
            kill_session(session, job_dir=job_dir)
            return 2
        window_logs[name] = str(window_log)

    meta_path = job_dir / "job.json"
    current = json.loads(meta_path.read_text(encoding="utf-8"))
    current["tmux_session"] = session
    current["window_logs"] = window_logs
    current["server_log"] = next((path for name, path in window_logs.items() if name.startswith("server")), None)
    current["client_log"] = next((path for name, path in window_logs.items() if name.startswith("client")), None)
    current["log_file"] = str(primary_log)
    meta_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")

    cprint(f"[green]Started eval tmux session:[/green] {session}")
    cprint(f"Job dir: {job_dir}")
    cprint(f"[cyan]Live monitor:[/cyan] bash interaction/bin/starvla-interact.sh monitor {job_dir.name}")
    if args.attach:
        tmux_attach(session)
    return 0


def _window_launch_delay(name: str, client_delay: int | float, stagger_seconds: int | float) -> float:
    stagger = max(float(stagger_seconds or 0.0), 0.0)
    match = re.search(r"(\d+)$", name)
    index = int(match.group(1)) if match else 0
    if name.startswith("server"):
        return index * stagger
    if name.startswith("client"):
        return max(float(client_delay or 0.0), 0.0) + index * stagger
    return max(float(client_delay or 0.0), 0.0)


def command_list(args: argparse.Namespace) -> int:
    rows = []
    for job in iter_jobs():
        latest = job.get("latest", {})
        metrics = latest.get("metrics") or {}
        loss_keys = [key for key in metrics if "loss" in key.lower() or key == "mse_score"]
        loss_text = " ".join(f"{_metric_label(key)}={compact_value(metrics[key])}" for key in loss_keys[:3])
        diagnosis = job.get("diagnosis") or {}
        severity = diagnosis.get("severity", "-")
        issue = (diagnosis.get("issues") or [{}])[0]
        rows.append(
            [
                Path(job.get("job_dir", "")).name,
                f"{job.get('kind', '-')} {status_markup('alive' if job.get('session_alive') else 'stopped')}",
                f"step={latest.get('step')}/{latest.get('max_steps') or '-'} ep={latest.get('epoch')} event={(job.get('latest_event') or {}).get('event', '-')}",
                f"{status_markup(severity)} {issue.get('code', '-')}",
                loss_text or "-",
            ]
        )
    print_table("Interaction jobs", ["job", "state", "progress", "health/root", "metrics"], rows)
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    job = find_job(args.job)
    if not job:
        cprint(f"[red]No job matched:[/red] {args.job}")
        return 1
    if getattr(args, "once", False):
        snapshot = collect_job_snapshot(job, tail_lines=args.tail_lines)
        console = rich_console()
        if console:
            console.print(render_snapshot(snapshot))
        else:
            print(snapshot_to_text(snapshot))
        return 0
    watch(job, interval=args.interval, tail_lines=args.tail_lines)
    return 0


def command_paths(args: argparse.Namespace) -> int:
    roots = artifact_roots_report()
    rows = [[key, value] for key, value in roots.items()]
    print_table("StarVLA artifact layout", ["item", "value"], rows)
    print_table(
        "Training save convention",
        ["artifact", "path"],
        [
            ["experiment group", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/"],
            ["model weights", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/checkpoints/steps_<N>_pytorch_model.pt"],
            ["top-k index", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/topk_checkpoints.json"],
            ["full resume state", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/accelerate_state/steps_<N>/"],
            ["latest resume state", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/accelerate_state/latest"],
            ["final model", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/final_model/"],
            ["training summary", f"{DEFAULT_CHECKPOINT_ROOT}/<model>-<action_head>-<dataset>/<run_id>/summary.jsonl"],
            ["tmux logs/events", f"{roots['active_runs_root']}/<timestamp>_<job>/"],
            ["H200 accelerate profile", "interaction/config/accelerate_h200_zero2.yaml"],
            ["H200 DeepSpeed profile", "interaction/config/ds_h200_zero2.json"],
        ],
    )
    return 0


def command_attach(args: argparse.Namespace) -> int:
    job = find_job(args.job)
    session = args.job
    if job:
        session = job.get("tmux_session", job.get("job_name", args.job))
    if not session_exists(session):
        cprint(f"[red]tmux session is not alive:[/red] {session}")
        return 1
    tmux_attach(session)
    return 0


def _proc_cmdline(pid: int) -> str:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return data.replace(b"\0", b" ").decode(errors="replace").strip()


def _proc_alive(pid: int) -> bool:
    if pid <= 1 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _eval_orphan_pids(job: dict[str, Any], job_dir: Path | None) -> set[int]:
    if job.get("kind") != "eval":
        return set()
    ckpt = str(job.get("ckpt") or "")
    ports = {str(item) for item in (job.get("ports") or []) if item is not None}
    if job.get("port") is not None:
        ports.add(str(job.get("port")))
    job_dir_text = str(job_dir.resolve()) if job_dir and job_dir.exists() else ""
    job_dir_name = job_dir.name if job_dir else ""
    pids: set[int] = set()
    proc_root = Path("/proc")
    if not proc_root.exists():
        return pids
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not _proc_alive(pid):
            continue
        cmd = _proc_cmdline(pid)
        if not cmd:
            continue
        is_server = "deployment/model_server/server_policy.py" in cmd
        is_calvin_client = "examples/calvin/eval_files/eval_calvin.py" in cmd
        if not (is_server or is_calvin_client):
            continue
        if job_dir_text and job_dir_text in cmd:
            pids.add(pid)
            continue
        if job_dir_name and job_dir_name in cmd:
            pids.add(pid)
            continue
        if ckpt and ckpt in cmd and (not ports or any(f"--port {port}" in cmd or f"--args.port {port}" in cmd for port in ports)):
            pids.add(pid)
            continue
        if ports and any(f"--port {port}" in cmd or f"--args.port {port}" in cmd for port in ports):
            pids.add(pid)
    return pids


def _terminate_pids(pids: set[int], *, grace_seconds: float = 1.5) -> int:
    targets = {pid for pid in pids if _proc_alive(pid)}
    if not targets:
        return 0
    for sig, delay in [(signal.SIGTERM, max(grace_seconds, 0.0)), (signal.SIGKILL, 0.0)]:
        for pid in sorted(targets, reverse=True):
            if _proc_alive(pid):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
        if delay:
            time.sleep(delay)
    return len({pid for pid in targets if not _proc_alive(pid)})


def command_stop(args: argparse.Namespace) -> int:
    job = find_job(args.job)
    session = args.job
    job_dir = None
    if job:
        session = job.get("tmux_session", job.get("job_name", args.job))
        if job.get("job_dir"):
            job_dir = Path(job["job_dir"])
    orphan_pids = _eval_orphan_pids(job or {}, job_dir)
    if orphan_pids:
        cprint(f"[yellow]Found eval GPU/client orphan processes before tmux cleanup:[/yellow] {len(orphan_pids)}")
        _terminate_pids(orphan_pids, grace_seconds=1.0)
    kill_session(session, job_dir=job_dir, grace_seconds=2.0)
    remaining = _eval_orphan_pids(job or {}, job_dir)
    if remaining:
        killed = _terminate_pids(remaining, grace_seconds=0.5)
        cprint(f"[yellow]Force-cleaned remaining eval processes:[/yellow] {killed}/{len(remaining)}")
    final_remaining = _eval_orphan_pids(job or {}, job_dir)
    if final_remaining:
        cprint(f"[red]Some eval processes are still alive:[/red] {sorted(final_remaining)[:20]}")
        return 1
    cprint(f"Stopped session and cleaned child processes if they existed: {session}")
    return 0


def command_selftest(args: argparse.Namespace) -> int:
    if not tmux_available():
        cprint("[red]tmux is required for selftest but is not available.[/red]")
        return 2

    session = sanitize_session(args.session or "starvla_interaction_selftest")
    if session_exists(session):
        kill_session(session)

    steps = max(args.steps, 1)
    command = f"""\
echo "[selftest] repo=$(pwd)"
echo "[selftest] python=$(command -v python)"
python - <<'PY'
import json
import os
import time

proxies = {{k: os.environ.get(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]}}
print("[selftest] proxies=" + json.dumps(proxies, sort_keys=True), flush=True)
if any(proxies.values()):
    raise SystemExit("proxy variables should be empty after activation")

for step in range({steps + 1}):
    metrics = {{
        "loss": round(1.0 / (step + 1), 6),
        "epoch": round(step / max({steps}, 1), 4),
        "mse_score": round(0.05 / (step + 1), 6),
        "timing/data": 0.001,
        "timing/model": 0.002,
    }}
    print(f"Step {{step}}, Loss: {{metrics}})", flush=True)
    time.sleep({args.sleep})

print("[selftest] done", flush=True)
PY
"""
    meta = {
        "kind": "selftest",
        "run_id": "selftest",
        "max_train_steps": steps,
        "tmux_session": session,
    }
    job_dir, runner, log_file = create_job_files("selftest", command, meta)
    meta_path = job_dir / "job.json"
    current = json.loads(meta_path.read_text(encoding="utf-8"))
    current["tmux_session"] = session
    meta_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")

    try:
        new_session(session, runner, REPO_ROOT, "selftest")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        _print_launch_failure("selftest tmux session creation", exc)
        return 2
    client_log = None
    if not args.single_window:
        client_command = """\
echo "[selftest-client] started"
python - <<'PY'
import time
time.sleep(0.2)
print("[selftest-client] second tmux window ok", flush=True)
PY
"""
        client_runner, client_log = create_extra_window_files(job_dir, "client-selftest", client_command)
        try:
            new_window(session, "client-test", client_runner, REPO_ROOT)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            _print_launch_failure("selftest client tmux window creation", exc)
            kill_session(session, job_dir=job_dir)
            return 2
        current = json.loads(meta_path.read_text(encoding="utf-8"))
        current["client_log"] = str(client_log)
        meta_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    cprint(f"[green]Started selftest tmux session:[/green] {session}")
    cprint(f"Job dir: {job_dir}")

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not session_exists(session):
            break
        time.sleep(0.5)

    if session_exists(session):
        cprint(f"[red]Selftest timed out; stopping tmux session:[/red] {session}")
        kill_session(session, job_dir=job_dir)
        return 1

    latest = parse_latest_metrics(log_file, steps)
    log_tail = "\n".join(tail(log_file, 20))
    print_table(
        "Selftest result",
        ["item", "value"],
        [
            ["job_dir", job_dir],
            ["log", log_file],
            ["latest_step", latest.get("step")],
            ["epoch", latest.get("epoch")],
            ["progress", latest.get("progress")],
            ["metrics", latest.get("metrics")],
        ],
    )
    if args.show_log:
        print(log_tail)

    client_ok = True
    if client_log is not None:
        client_tail = "\n".join(tail(client_log, 20))
        client_ok = "[selftest-client] second tmux window ok" in client_tail
        if args.show_log:
            print(client_tail)

    if latest.get("step") != steps or "[selftest] done" not in log_tail:
        cprint("[red]Selftest failed: expected final step/log marker was not found.[/red]")
        return 1
    if not client_ok:
        cprint("[red]Selftest failed: second tmux window marker was not found.[/red]")
        return 1
    cprint("[green]Selftest passed.[/green]")
    return 0


def _rich_panel(title: str, message: str, style: str = "cyan") -> None:
    rich = rich_console()
    if rich:
        from rich.panel import Panel

        rich.print(Panel(message, title=title, border_style=style))
    else:
        print(f"\n{title}\n{remove_markup(message)}")


def _parse_optional_int(value: str | None, *, name: str, min_value: int | None = None) -> int | None:
    text = "" if value is None else str(value).strip().lower()
    if text in {"", "auto", "none", "default", "preset"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or auto") from exc
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return parsed


def _ask_optional_int(prompt: str, *, name: str, min_value: int | None = None) -> int | None:
    while True:
        raw = input(prompt).strip()
        try:
            return _parse_optional_int(raw, name=name, min_value=min_value)
        except ValueError as exc:
            cprint(f"[red]Invalid input:[/red] {exc}. Use a number or [cyan]auto[/cyan].")


def _ask_choice(title: str, values: list[str], labels: dict[str, str] | None = None, default: int = 0) -> str:
    labels = labels or {}
    if not values:
        raise ValueError(f"No choices available for {title}")
    default = max(0, min(default, len(values) - 1))
    while True:
        rich = rich_console()
        if rich:
            from rich import box
            from rich.table import Table

            table = Table(title=title, box=box.ROUNDED, show_lines=False, expand=False)
            table.add_column("#", style="bold cyan", justify="right", no_wrap=True)
            table.add_column("choice", style="bold white", no_wrap=True)
            table.add_column("detail", style="dim", overflow="fold")
            for i, value in enumerate(values, start=1):
                default_marker = " [green]default[/green]" if i - 1 == default else ""
                table.add_row(str(i), f"{value}{default_marker}", labels.get(value, value))
            rich.print(table)
        else:
            print(f"\n{title}")
            for i, value in enumerate(values, start=1):
                suffix = " [default]" if i - 1 == default else ""
                print(f"  {i}. {value} - {labels.get(value, value)}{suffix}")
        raw = input("> ").strip()
        if not raw:
            return values[default]
        if raw in values:
            return raw
        try:
            index = int(raw) - 1
        except ValueError:
            cprint(f"[red]Invalid choice:[/red] {raw}. Enter a number or one of: {', '.join(values)}")
            continue
        if 0 <= index < len(values):
            return values[index]
        cprint(f"[red]Invalid choice index:[/red] {raw}. Choose 1-{len(values)}.")


def _available_train_presets(catalog: dict[str, Any]) -> list[str]:
    available: list[str] = []
    for key, preset in catalog.get("train_presets", {}).items():
        data_root = preset.get("data_root")
        if data_root and not bool((resolve_repo_path(data_root) or Path(data_root)).exists()):
            continue
        available.append(key)
    return available


def command_menu(args: argparse.Namespace) -> int:
    while True:
        choice = _ask_choice(
            "StarVLA interaction",
            ["check", "catalog", "perf_check", "paths", "selftest", "launch_train", "launch_eval", "list", "monitor", "attach", "stop", "exit"],
            {
                "check": "environment, paths, imports, action heads",
                "catalog": "datasets, base models, action experts, training/structure policies",
                "perf_check": "GPU topology, NCCL profile, bandwidth-test readiness",
                "paths": "show checkpoint and interaction run save locations",
                "selftest": "real tmux CPU-only orchestration test",
                "launch_train": "start a tmux-managed training job",
                "launch_eval": "start a tmux-managed websocket evaluation",
                "list": "show managed jobs",
                "monitor": "live log and metric dashboard",
                "attach": "attach to a tmux session",
                "stop": "stop a tmux session",
                "exit": "quit",
            },
        )
        if choice == "exit":
            return 0
        if choice == "check":
            command_check(argparse.Namespace(skip_action_heads=False))
        elif choice == "catalog":
            command_catalog(argparse.Namespace(kind="all"))
        elif choice == "perf_check":
            command_perf_check(
                argparse.Namespace(
                    gpus="auto",
                    num_gpus=8,
                    min_free_mb=60000,
                    perf_profile="auto",
                    nccl_debug="WARN",
                    nccl_trace=False,
                    nccl_socket_ifname=None,
                    nccl_ib_hca=None,
                    nccl_p2p_level="auto",
                    nccl_env=[],
                    run_nccl_test=False,
                    nccl_test_max_bytes="16G",
                    timeout=300,
                )
            )
        elif choice == "paths":
            command_paths(argparse.Namespace())
        elif choice == "selftest":
            command_selftest(argparse.Namespace(session=None, steps=3, sleep=0.2, timeout=30, show_log=True, single_window=False))
        elif choice == "list":
            command_list(argparse.Namespace())
        elif choice == "monitor":
            command_monitor(argparse.Namespace(job=input("job name or run dir: ").strip(), interval=2.0, tail_lines=18, once=False))
        elif choice == "attach":
            command_attach(argparse.Namespace(job=input("job name or run dir: ").strip()))
        elif choice == "stop":
            command_stop(argparse.Namespace(job=input("job name or run dir: ").strip()))
        elif choice == "launch_train":
            catalog = load_catalog()
            policy_catalog = load_policy_catalog()
            dataset_values = policy_keys("datasets")
            dataset_key = _ask_choice("Dataset", dataset_values, {k: v.get("label", k) for k, v in policy_catalog.get("datasets", {}).items()})
            preset = policy_catalog["datasets"][dataset_key].get("preset", "libero_vla")
            action_values = all_action_expert_keys()
            default_action = catalog["train_presets"][preset].get("framework", "PI")
            action_expert = _ask_choice("Action expert", action_values, {k: policy_catalog.get("action_experts", {}).get(k, catalog.get("frameworks", {}).get(k, {})).get("label", k) for k in action_values}, default=max(0, action_values.index(default_action) if default_action in action_values else 0))
            default_model = catalog["train_presets"][preset].get("model") or default_model_for_framework(action_expert)
            model_values = all_model_keys()
            model = _ask_choice("Base model", model_values, {k: policy_catalog.get("base_models", {}).get(k, catalog.get("models", {}).get(k, {})).get("label", k) for k in model_values}, default=max(0, model_values.index(default_model) if default_model in model_values else 0))
            training_default = policy_catalog["datasets"][dataset_key].get("recommended_training_policies", [])[:3]
            structure_default = policy_catalog["datasets"][dataset_key].get("recommended_structure_policies", [])[:3]
            training_policies = input(f"Training policies comma/space list [{','.join(training_default)}]: ").strip()
            structure_policies = input(f"Structure policies comma/space list [{','.join(structure_default)}]: ").strip()
            training_policy_values = split_multi_values([training_policies]) or training_default
            structure_policy_values = split_multi_values([structure_policies]) or structure_default
            framework, _ = resolve_action_expert(action_expert, base_model_key(model))
            preset_cfg = catalog["train_presets"][preset]
            dataset = dataset_key
            experiment_group = experiment_group_name(model, framework, dataset)
            combo_slug = policy_combo_slug(training_policy_values, structure_policy_values)
            if combo_slug:
                experiment_group = f"{experiment_group}-{combo_slug}"
            checkpoint_root = preset_cfg.get("run_root_dir", str(DEFAULT_CHECKPOINT_ROOT))
            group_root = Path(checkpoint_root) / experiment_group
            latest_run = latest_checkpoint_run_id(preset_cfg.get("run_id_prefix", preset), group_root)
            print_table(
                "Training Context",
                ["item", "value"],
                [
                    ["preset", preset],
                    ["dataset", dataset_key],
                    ["training_policy", ",".join(training_policy_values) or "-"],
                    ["base_model", model],
                    ["action_expert", action_expert],
                    ["resolved_framework", framework],
                    ["structure_policy", ",".join(structure_policy_values) or "-"],
                    ["experiment_group", experiment_group],
                    ["checkpoint_group_root", group_root],
                    ["latest_run", latest_run or "-"],
                ],
                styles=["cyan", ""],
            )
            run_mode = _ask_choice(
                "Run mode",
                ["continue", "new"],
                {
                    "continue": f"resume latest/current run under {experiment_group}" + (f" (latest: {latest_run})" if latest_run else " (no run found yet)"),
                    "new": f"create a new run under {experiment_group}",
                },
                default=0 if latest_run else 1,
            )
            resume_mode = "auto"
            resume_from_checkpoint = None
            if run_mode == "continue":
                resume_mode = _ask_choice(
                    "Resume mode",
                    ["auto", "full", "weights"],
                    {
                        "auto": "prefer full Accelerate state, fall back to weights",
                        "full": "require optimizer/scheduler/RNG state",
                        "weights": "load model weights only",
                    },
                )
                resume_from_checkpoint = normalize_run_id(input("Explicit checkpoint/state path [auto]: ").strip())
            init_checkpoint = None
            reload_modules = None
            if run_mode == "new":
                init_checkpoint = normalize_run_id(input("Post-train from checkpoint [none]: ").strip())
                if init_checkpoint:
                    reload_modules = normalize_run_id(input("Reload modules from checkpoint [full model]: ").strip())
            gpu_request = input("GPUs (auto/all/0,1...) [auto]: ").strip() or "auto"
            num_gpus = _ask_num_gpus()
            if run_mode == "continue":
                run_id_prompt = f"run_id [latest: {latest_run or 'auto'}]: "
            else:
                run_id_prompt = "run_id [auto]: "
            run_id = normalize_run_id(input(run_id_prompt).strip())
            max_steps = _ask_optional_int("max_train_steps [preset]: ", name="max_train_steps", min_value=1)
            throughput_profile = _ask_choice(
                "Throughput profile",
                ["auto", "h200_saturated", "balanced", "conservative", "none"],
                {
                    "auto": "detect H200 and choose an appropriate profile",
                    "h200_saturated": "aggressive H200 batch/dataloader profile",
                    "balanced": "moderate batch/dataloader profile",
                    "conservative": "small safe profile",
                    "none": "do not override batch/dataloader settings",
                },
            )
            _rich_panel(
                "Throughput Inputs",
                "[cyan]auto[/cyan] lets the selected throughput profile choose values. "
                "For H200, start with auto or a moderate explicit batch. "
                "FAST/QwenFast action-token runs should use auto or 8/16; "
                "if OOM occurs, relaunch with a smaller batch and [bold]Run mode = continue[/bold].",
                style="magenta",
            )
            vla_batch_size = _ask_optional_int("VLA per-device batch [auto]: ", name="VLA per-device batch", min_value=1)
            vlm_batch_size = _ask_optional_int("VLM per-device batch [auto]: ", name="VLM per-device batch", min_value=1)
            dataloader_workers = _ask_optional_int("DataLoader workers per rank [auto]: ", name="DataLoader workers", min_value=0)
            ns = argparse.Namespace(
                preset=preset,
                dataset=dataset_key,
                framework=None,
                action_expert=action_expert,
                model=None,
                base_model=model,
                training_policy=training_policy_values,
                structure_policy=structure_policy_values,
                gpus=gpu_request,
                num_gpus=num_gpus,
                min_free_mb=60000,
                run_id=run_id,
                max_steps=max_steps,
                vla_batch_size=vla_batch_size,
                vlm_batch_size=vlm_batch_size,
                dataloader_workers=dataloader_workers,
                dataloader_prefetch_factor=None,
                dataloader_pin_memory=None,
                dataloader_persistent_workers=None,
                data_root=None,
                data_mix=None,
                vlm_dataset=None,
                run_root_dir=None,
                experiment_group=experiment_group,
                run_mode=run_mode,
                resume=(run_mode == "continue"),
                resume_from_checkpoint=resume_from_checkpoint,
                resume_mode=resume_mode,
                init_checkpoint=init_checkpoint,
                reload_modules=reload_modules,
                no_full_state=False,
                keep_top_k=3,
                freeze_modules=None,
                mixed_precision="bf16",
                num_machines=1,
                machine_rank=0,
                main_process_ip=None,
                main_process_port=None,
                allow_readonly_output=False,
                allow_cpu=False,
                perf_profile="auto",
                nccl_debug="WARN",
                nccl_trace=False,
                nccl_socket_ifname=None,
                nccl_ib_hca=None,
                nccl_p2p_level="auto",
                nccl_env=[],
                throughput_profile=throughput_profile,
                enable_wandb=False,
                extra=[],
                dry_run=False,
                session=None,
                attach=False,
                yes=False,
            )
            command_train(ns)
        elif choice == "launch_eval":
            catalog = load_catalog()
            preset = _ask_choice("Eval preset", keys("eval_presets"), {k: v["label"] for k, v in catalog["eval_presets"].items()})
            ckpt = input("checkpoint path: ").strip()
            trials = None
            max_steps_per_task = None
            parallel_workers = None
            replicas_per_gpu = None
            eval_throughput_profile = "auto"
            calvin_render_backend = "auto"
            if preset == "calvin":
                _rich_panel(
                    "CALVIN Evaluation",
                    "The official CALVIN long-horizon list has 1000 sequences. "
                    "Use [cyan]official[/cyan] for the benchmark score, or [cyan]quick[/cyan]/[cyan]medium[/cyan] for fast iteration. "
                    "Set server GPU(s) to [bold]all[/bold] or a comma list to shard sequences across multiple policy servers.",
                    style="cyan",
                )
                scale = _ask_choice(
                    "Evaluation Scale",
                    ["quick", "medium", "official", "custom"],
                    {
                        "quick": "20 sequences, 120 max steps per task",
                        "medium": "100 sequences, 360 official max steps per task",
                        "official": "1000 official CALVIN sequences, 360 max steps per task",
                        "custom": "enter sequence count and max steps manually",
                    },
                    default=1,
                )
                if scale == "quick":
                    trials, max_steps_per_task = 20, 120
                elif scale == "medium":
                    trials, max_steps_per_task = 100, 360
                elif scale == "official":
                    trials, max_steps_per_task = 1000, 360
                else:
                    trials = _ask_optional_int("CALVIN sequences [1000]: ", name="CALVIN sequences", min_value=1) or 1000
                    max_steps_per_task = _ask_optional_int("Max steps per task [360]: ", name="max steps per task", min_value=1) or 360
                server_gpu = input("server GPU(s) [auto/all/0,1...] [auto]: ").strip() or "auto"
                eval_throughput_profile = _ask_choice(
                    "Eval Throughput Profile",
                    ["auto", "h200_saturated", "balanced", "conservative"],
                    {
                        "auto": "default multi-replica profile; uses more than one server per H200",
                        "h200_saturated": "aggressive; higher GPU memory use and more simulator workers",
                        "balanced": "2 policy-server replicas per selected GPU",
                        "conservative": "1 policy-server replica per selected GPU",
                    },
                )
                replicas_per_gpu = _ask_optional_int("Policy server replicas per GPU [profile]: ", name="replicas per GPU", min_value=1)
                parallel_workers = _ask_optional_int("Total eval workers [auto]: ", name="parallel eval workers", min_value=1)
                calvin_render_backend = _ask_choice(
                    "CALVIN Render Backend",
                    ["auto", "egl", "direct"],
                    {
                        "auto": "try EGL GPU headless rendering and fall back to direct/osmesa if preflight fails",
                        "egl": "force fast GPU headless rendering",
                        "direct": "safe CPU/software rendering fallback",
                    },
                )
                client_gpu = ""
            else:
                server_gpu = input("server GPU [0]: ").strip() or "0"
                client_gpu = input("client GPU [0]: ").strip() or "0"
            ns = argparse.Namespace(
                preset=preset,
                ckpt=ckpt,
                server_gpu=server_gpu,
                client_gpu=client_gpu,
                port=None,
                server_only=False,
                task=None,
                trials=trials,
                max_steps_per_task=max_steps_per_task,
                parallel_workers=parallel_workers,
                replicas_per_gpu=replicas_per_gpu,
                eval_throughput_profile=eval_throughput_profile,
                calvin_render_backend=calvin_render_backend,
                allow_many_workers=False,
                dataset_path=None,
                eval_log_dir=None,
                extra=[],
                dry_run=False,
                session=None,
                client_delay=8,
                launch_stagger_seconds=0.5,
                attach=False,
                yes=False,
            )
            command_eval(ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified interactive StarVLA orchestration.")
    sub = parser.add_subparsers(dest="command", required=False)

    selftest = sub.add_parser("selftest", help="run a real tmux CPU-only orchestration self-test")
    selftest.add_argument("--session")
    selftest.add_argument("--steps", type=int, default=3)
    selftest.add_argument("--sleep", type=float, default=0.2)
    selftest.add_argument("--timeout", type=float, default=30)
    selftest.add_argument("--show-log", action="store_true")
    selftest.add_argument("--single-window", action="store_true", help="skip the second tmux window check")
    selftest.set_defaults(func=command_selftest)

    check = sub.add_parser("check", help="run environment, model, action-head, and path checks")
    check.add_argument("--skip-action-heads", action="store_true")
    check.set_defaults(func=command_check)

    paths = sub.add_parser("paths", help="show checkpoint, run, and tmux artifact locations")
    paths.set_defaults(func=command_paths)

    cat = sub.add_parser("catalog", help="show selectable datasets, base models, action experts, and T/S policies")
    cat.add_argument(
        "--kind",
        choices=["all", "datasets", "base-models", "models", "action-experts", "experts", "training-policies", "training", "structure-policies", "structure"],
        default="all",
    )
    cat.set_defaults(func=command_catalog)

    perf = sub.add_parser("perf-check", help="inspect GPU topology, NCCL profile, and bandwidth-test readiness")
    perf.add_argument("--gpus", default="auto", help="auto, all, or comma-separated GPU ids")
    perf.add_argument("--num-gpus", type=parse_num_gpus, default=8, help="integer GPU count, all, or auto")
    perf.add_argument("--min-free-mb", type=int, default=60000)
    perf.add_argument("--perf-profile", choices=["auto", "balanced", "h200_8gpu", "debug", "none"], default="auto")
    perf.add_argument("--nccl-debug", choices=["WARN", "INFO", "VERSION"], default="WARN")
    perf.add_argument("--nccl-trace", action="store_true")
    perf.add_argument("--nccl-socket-ifname")
    perf.add_argument("--nccl-ib-hca")
    perf.add_argument("--nccl-p2p-level", choices=["auto", "LOC", "NVL", "PIX", "PXB", "PHB", "SYS"], default="auto")
    perf.add_argument("--nccl-env", nargs="*", default=[])
    perf.add_argument("--run-nccl-test", action="store_true", help="run all_reduce_perf if NVIDIA nccl-tests is installed")
    perf.add_argument("--nccl-test-max-bytes", default="16G")
    perf.add_argument("--timeout", type=int, default=300)
    perf.set_defaults(func=command_perf_check)

    train = sub.add_parser("train", help="launch tmux-managed training")
    train.add_argument("--preset", choices=keys("train_presets"), default="libero_vla")
    train.add_argument("--dataset", choices=policy_keys("datasets"), help="benchmark dataset id; overrides the preset data root/mix")
    train.add_argument("--framework", choices=all_action_expert_keys(), help="legacy concrete framework/action head id")
    train.add_argument("--action-expert", choices=all_action_expert_keys(), help="free action expert id; family-routed aliases include PI/OFT/GR00T/FAST")
    train.add_argument("--model", help="model key from catalog or explicit path")
    train.add_argument("--base-model", choices=all_model_keys(), help="free base model id; alias for --model with policy metadata")
    train.add_argument("--training-policy", "--training-policies", nargs="+", action="append", default=[], help="training policy IDs, e.g. T01 T06 T07 or T01,T06")
    train.add_argument("--structure-policy", "--structure-policies", nargs="+", action="append", default=[], help="structure policy IDs, e.g. S01 S07")
    train.add_argument("--gpus", default="auto", help="auto, all, or comma-separated GPU ids")
    train.add_argument("--num-gpus", type=parse_num_gpus, help="integer GPU count, all, or auto")
    train.add_argument("--min-free-mb", type=int, default=60000)
    train.add_argument("--run-id")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--vla-batch-size", type=int)
    train.add_argument("--vlm-batch-size", type=int)
    train.add_argument("--data-root")
    train.add_argument("--data-mix")
    train.add_argument("--vlm-dataset")
    train.add_argument("--run-root-dir")
    train.add_argument("--experiment-group", help="override the <model>-<action_head>-<dataset> group directory name")
    train.add_argument("--run-mode", choices=["new", "continue"], default="new", help="new creates a new run_id; continue resumes an existing run_id")
    train.add_argument("--resume", action="store_true", help="resume from the latest checkpoint under the run output directory")
    train.add_argument("--resume-from-checkpoint", help="explicit Accelerate state directory or model weight file to resume from")
    train.add_argument("--resume-mode", choices=["auto", "full", "weights"], default="auto")
    train.add_argument("--init-checkpoint", "--posttrain-from", dest="init_checkpoint", help="load an existing trained model as initialization for a new post-training run")
    train.add_argument("--reload-modules", help="comma-separated module paths to load from --init-checkpoint, e.g. action_model,qwen_vl_interface")
    train.add_argument("--no-full-state", action="store_true", help="only save model weights; disables optimizer/RNG/scheduler state snapshots")
    train.add_argument("--keep-top-k", type=int, default=3, help="keep only the best K model-weight checkpoints by lowest loss")
    train.add_argument("--freeze-modules", help="comma-separated module paths to freeze; default is backbone for CosmoPredict2 and empty for Qwen")
    train.add_argument("--mixed-precision", choices=["bf16", "fp16", "no", "config"], default="bf16")
    train.add_argument("--num-machines", type=int, default=1)
    train.add_argument("--machine-rank", type=int, default=0)
    train.add_argument("--main-process-ip")
    train.add_argument("--main-process-port", type=int)
    train.add_argument("--allow-readonly-output", action="store_true", help="launch even if the current shell cannot write --run-root-dir")
    train.add_argument("--allow-cpu", action="store_true", help="launch without a selected GPU; intended only for tiny smoke tests")
    train.add_argument("--perf-profile", choices=["auto", "balanced", "h200_8gpu", "debug", "none"], default="auto", help="NCCL/DeepSpeed performance profile")
    train.add_argument("--nccl-debug", choices=["WARN", "INFO", "VERSION"], default="WARN", help="NCCL_DEBUG level used by the runner")
    train.add_argument("--nccl-trace", action="store_true", help="enable PyTorch NCCL flight-recorder dump on timeout")
    train.add_argument("--nccl-socket-ifname", help="manual NCCL_SOCKET_IFNAME, e.g. '=ib0' or '^docker'")
    train.add_argument("--nccl-ib-hca", help="manual NCCL_IB_HCA, e.g. '=mlx5_0:1,=mlx5_1:1'")
    train.add_argument("--nccl-p2p-level", choices=["auto", "LOC", "NVL", "PIX", "PXB", "PHB", "SYS"], default="auto")
    train.add_argument("--nccl-env", nargs="*", default=[], help="force extra NCCL/TORCH_NCCL env overrides as KEY=VALUE")
    train.add_argument("--throughput-profile", choices=["auto", "h200_saturated", "balanced", "conservative", "none"], default="auto", help="batch/dataloader/CPU-thread profile for GPU saturation")
    train.add_argument("--dataloader-workers", type=int, help="DataLoader workers per rank; auto profile uses 8 on H200")
    train.add_argument("--dataloader-prefetch-factor", type=int, help="DataLoader prefetch factor per worker")
    pin = train.add_mutually_exclusive_group()
    pin.add_argument("--pin-memory", dest="dataloader_pin_memory", action="store_true")
    pin.add_argument("--no-pin-memory", dest="dataloader_pin_memory", action="store_false")
    train.set_defaults(dataloader_pin_memory=None)
    persistent = train.add_mutually_exclusive_group()
    persistent.add_argument("--persistent-workers", dest="dataloader_persistent_workers", action="store_true")
    persistent.add_argument("--no-persistent-workers", dest="dataloader_persistent_workers", action="store_false")
    train.set_defaults(dataloader_persistent_workers=None)
    train.add_argument("--enable-wandb", action="store_true")
    train.add_argument("--extra", nargs="*", default=[])
    train.add_argument("--session")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--attach", action="store_true")
    train.add_argument("--yes", action="store_true", help="skip the interactive launch confirmation")
    train.set_defaults(func=command_train)

    ev = sub.add_parser("eval", help="launch tmux-managed evaluation")
    ev.add_argument("--preset", choices=keys("eval_presets"), default="calvin")
    ev.add_argument("--ckpt")
    ev.add_argument("--server-gpu", default="0")
    ev.add_argument("--client-gpu", default="0")
    ev.add_argument("--port", type=int)
    ev.add_argument("--server-only", action="store_true")
    ev.add_argument("--task")
    ev.add_argument("--trials", type=int)
    ev.add_argument("--max-steps-per-task", type=int, help="CALVIN max simulation steps per subtask; official setting is 360")
    ev.add_argument("--parallel-workers", type=int, help="CALVIN total server/client worker count; default is selected GPUs times the eval throughput profile replicas")
    ev.add_argument("--replicas-per-gpu", type=int, help="CALVIN policy-server replicas per selected GPU; ignored when --parallel-workers is set")
    ev.add_argument("--allow-many-workers", action="store_true", help="allow more than 64 CALVIN eval workers; use only after checking system limits")
    ev.add_argument(
        "--eval-throughput-profile",
        choices=["auto", "h200_saturated", "balanced", "conservative"],
        default="auto",
        help="CALVIN eval concurrency profile; auto uses multiple policy-server replicas per GPU",
    )
    ev.add_argument(
        "--calvin-render-backend",
        choices=["auto", "egl", "direct", "osmesa", "safe", "cpu", "fast", "gpu", "cuda"],
        default="auto",
        help="CALVIN rendering backend: auto tries EGL GPU headless rendering, direct/osmesa uses the safe CPU path",
    )
    ev.add_argument("--dataset-path", help="CALVIN dataset path override; defaults to the preset D-D validation path")
    ev.add_argument("--eval-log-dir", help="CALVIN evaluation log directory; defaults to the tmux job directory")
    ev.add_argument("--client-delay", type=int, default=8)
    ev.add_argument("--launch-stagger-seconds", type=float, default=0.5, help="stagger eval worker window startup to avoid process storms")
    ev.add_argument("--extra", nargs="*", default=[])
    ev.add_argument("--session")
    ev.add_argument("--dry-run", action="store_true")
    ev.add_argument("--attach", action="store_true")
    ev.add_argument("--yes", action="store_true", help="skip the interactive launch confirmation")
    ev.set_defaults(func=command_eval)

    jobs = sub.add_parser("list", help="list managed jobs")
    jobs.set_defaults(func=command_list)

    mon = sub.add_parser("monitor", help="live monitor a managed job")
    mon.add_argument("job")
    mon.add_argument("--interval", type=float, default=2.0)
    mon.add_argument("--tail-lines", type=int, default=18, help="recent log lines per managed window")
    mon.add_argument("--once", action="store_true", help="print one metrics snapshot and exit")
    mon.set_defaults(func=command_monitor)

    att = sub.add_parser("attach", help="attach to a managed tmux session")
    att.add_argument("job")
    att.set_defaults(func=command_attach)

    stop = sub.add_parser("stop", help="stop a managed tmux session")
    stop.add_argument("job")
    stop.set_defaults(func=command_stop)

    menu = sub.add_parser("menu", help="open interactive menu")
    menu.set_defaults(func=command_menu)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args = parser.parse_args(["menu"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
