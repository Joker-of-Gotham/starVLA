from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .catalog import load_catalog, resolve_repo_path
from .gpu import detect_gpus
from .paths import ACTIVATE_SCRIPT, DEFAULT_PYTHON, REPO_ROOT, artifact_roots_report
from .tmux import tmux_available


def _run_bash(script: str, timeout: int = 120) -> tuple[int, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    proc = subprocess.run(["bash", "-lc", script], cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout, env=env)
    return proc.returncode, proc.stdout + proc.stderr


def _extract_marked_json(output: str, marker: str = "STARVLA_CHECK_JSON=") -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        return json.loads(payload)
    for line in reversed(output.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            parsed = json.loads(text)
            if "imports" in parsed and "proxies" in parsed:
                return parsed
    raise ValueError(f"missing {marker!r} line")


def environment_report(include_action_heads: bool = True) -> dict[str, Any]:
    catalog = load_catalog()
    report: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "activate_script": ACTIVATE_SCRIPT.exists(),
        "python": str(DEFAULT_PYTHON),
        "python_exists": DEFAULT_PYTHON.exists(),
        "tmux": tmux_available(),
        "gpus": [gpu.__dict__ | {"memory_free_mb": gpu.memory_free_mb} for gpu in detect_gpus()],
        "required_paths": [],
        "imports": {},
        "runtime_capabilities": {},
        "proxy_clean": None,
        "action_heads": None,
        "artifacts": artifact_roots_report(),
    }

    for item in catalog.get("required_paths", []):
        path = resolve_repo_path(item)
        report["required_paths"].append({"path": item, "exists": bool(path and path.exists())})

    if ACTIVATE_SCRIPT.exists() and DEFAULT_PYTHON.exists():
        code = r"""
source bar/activate_starvla_no_proxy.sh
python - <<'PY'
import json, os
from importlib.metadata import PackageNotFoundError, version
mods = {}
try:
    import torch
    cuda = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    torch_cuda_version = torch.version.cuda
except Exception:
    cuda = False
    cuda_device_count = 0
    torch_cuda_version = None
for name in ["torch", "transformers", "accelerate", "diffusers", "decord", "pytorch3d", "omegaconf", "rich", "triton"]:
    try:
        mod = __import__(name)
        mods[name] = getattr(mod, "__version__", "ok")
    except Exception as exc:
        mods[name] = "ERROR: " + repr(exc)
if cuda:
    try:
        mod = __import__("deepspeed")
        mods["deepspeed"] = getattr(mod, "__version__", "ok")
    except Exception as exc:
        mods["deepspeed"] = "ERROR: " + repr(exc)
else:
    try:
        mods["deepspeed"] = "installed " + version("deepspeed") + " (metadata; CUDA unavailable)"
    except PackageNotFoundError as exc:
        mods["deepspeed"] = "ERROR: " + repr(exc)
for name in ["flash_attn"]:
    try:
        mod = __import__(name)
        mods[name] = getattr(mod, "__version__", "ok")
    except Exception as exc:
        mods[name] = "OPTIONAL_MISSING: " + repr(exc)
capabilities = {}
try:
    import transformers

    def check_transformers_attr(label, attr):
        try:
            getattr(transformers, attr)
            capabilities[label] = "ok " + attr
        except Exception as exc:
            cause = getattr(exc, "__cause__", None)
            detail = repr(exc)
            if cause is not None:
                detail += "; cause=" + repr(cause)
            capabilities[label] = "ERROR: " + detail

    check_transformers_attr("Qwen3.5", "Qwen3_5ForConditionalGeneration")
    check_transformers_attr("Qwen3-VL", "Qwen3VLForConditionalGeneration")
except Exception as exc:
    capabilities["transformers_runtime"] = "ERROR: " + repr(exc)
proxies = {k: os.environ.get(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]}
print("STARVLA_CHECK_JSON=" + json.dumps({
    "imports": mods,
    "runtime_capabilities": capabilities,
    "proxies": proxies,
    "cuda_available": cuda,
    "cuda_device_count": cuda_device_count,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_version": torch_cuda_version,
}, sort_keys=True))
PY
"""
        rc, out = _run_bash(code)
        report["import_check_rc"] = rc
        report["import_check_output"] = out[-4000:]
        try:
            parsed = _extract_marked_json(out)
            report["imports"] = parsed["imports"]
            report["runtime_capabilities"] = parsed.get("runtime_capabilities", {})
            report["proxy_clean"] = all(value in {None, ""} for value in parsed["proxies"].values())
            report["cuda_available"] = parsed["cuda_available"]
            report["cuda_device_count"] = parsed.get("cuda_device_count")
            report["cuda_visible_devices"] = parsed.get("cuda_visible_devices")
            report["torch_cuda_version"] = parsed.get("torch_cuda_version")
            if rc != 0:
                report["imports"]["import_check"] = "ERROR: import subprocess exited " + str(rc)
        except Exception as exc:
            report["imports"] = {"import_check": "ERROR: unable to parse import subprocess output: " + repr(exc)}
            report["runtime_capabilities"] = {}
            report["proxy_clean"] = False
            report["import_check_parse_error"] = repr(exc)

    if include_action_heads and (REPO_ROOT / "bar" / "check_action_heads.py").exists():
        rc, out = _run_bash("source bar/activate_starvla_no_proxy.sh && python bar/check_action_heads.py --extended", timeout=180)
        report["action_heads"] = {"returncode": rc, "output": out[-4000:]}

    return report


def model_quick_report() -> dict[str, Any]:
    catalog = load_catalog()
    out: dict[str, Any] = {}
    for key, model in catalog.get("models", {}).items():
        model_path = model.get("path")
        path = resolve_repo_path(model_path)
        out[key] = {
            "label": model.get("label", key),
            "path": model_path,
            "exists": bool(path and path.exists()),
        }
    return out


def dataset_quick_report(include_optional_missing: bool = False) -> dict[str, Any]:
    catalog = load_catalog()
    out: dict[str, Any] = {}
    for key, preset in catalog.get("train_presets", {}).items():
        data_root = preset.get("data_root")
        if not data_root:
            continue
        path = resolve_repo_path(data_root)
        exists = bool(path and path.exists())
        optional = bool(preset.get("optional", False))
        if optional and not exists and not include_optional_missing:
            continue
        out[key] = {
            "label": preset.get("label", key),
            "data_root": data_root,
            "exists": exists,
            "optional": optional,
            "data_mix": preset.get("data_mix"),
        }
    return out
