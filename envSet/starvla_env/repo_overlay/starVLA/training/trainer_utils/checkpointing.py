from __future__ import annotations

import os
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"steps_(\d+)")


def cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def cfg_bool(obj: Any, name: str, default: bool = False) -> bool:
    value = cfg_get(obj, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def cfg_str(obj: Any, name: str, default: str | None = None) -> str | None:
    value = cfg_get(obj, name, default)
    if value is None:
        return default
    return str(value)


def parse_step_from_path(path: str | os.PathLike[str] | None) -> int:
    if not path:
        return 0
    match = STEP_RE.search(str(path))
    return int(match.group(1)) if match else 0


def latest_state_checkpoint(state_root: str | os.PathLike[str]) -> tuple[str | None, int]:
    root = Path(state_root)
    if not root.exists():
        return None, 0
    latest = root / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if resolved.exists() and resolved.is_dir():
            return str(latest), parse_step_from_path(resolved)
    candidates: list[tuple[Path, int]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        step = parse_step_from_path(path)
        if step > 0:
            candidates.append((path, step))
    if not candidates:
        return None, 0
    path, step = sorted(candidates, key=lambda item: item[1])[-1]
    return str(path), step


def checkpoint_summary(
    *,
    completed_steps: int,
    weight_checkpoint: str | None,
    state_checkpoint: str | None,
    save_format: str,
    resume_mode: str,
) -> dict[str, Any]:
    return {
        "steps": completed_steps,
        "weight_checkpoint": weight_checkpoint,
        "state_checkpoint": state_checkpoint,
        "save_format": save_format,
        "resume_mode": resume_mode,
    }


def checkpoint_loss(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    if _finite_number(metrics.get("loss")):
        return float(metrics["loss"])
    excluded_prefixes = ("adaptive_lr/", "timing/")
    excluded_keys = {"nonfinite_loss_skipped"}
    loss_values = [
        float(value)
        for key, value in metrics.items()
        if key not in excluded_keys
        and not key.startswith(excluded_prefixes)
        and _finite_number(value)
        and (key == "loss" or key.endswith("_loss") or key.endswith("/loss") or "loss" in key.lower())
    ]
    if loss_values:
        return float(sum(loss_values))
    if _finite_number(metrics.get("mse_score")):
        return float(metrics["mse_score"])
    return None


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}


def update_latest_state_link(state_root: str | os.PathLike[str], state_checkpoint: str | None) -> str | None:
    if not state_checkpoint:
        return None
    root = Path(state_root)
    target = Path(state_checkpoint)
    latest = root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(target.name, target_is_directory=True)
        return str(latest)
    except OSError:
        marker = root / "latest_state.txt"
        marker.write_text(str(target) + "\n", encoding="utf-8")
        return str(marker)


def update_topk_checkpoints(
    *,
    output_dir: str | os.PathLike[str],
    completed_steps: int,
    weight_checkpoint: str | None,
    state_checkpoint: str | None,
    metrics: dict[str, Any] | None,
    k: int = 3,
    keep_latest_state: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    index_path = output / "topk_checkpoints.json"
    topk = _load_topk(index_path)
    loss = checkpoint_loss(metrics)

    if weight_checkpoint and loss is not None:
        candidate = {
            "steps": completed_steps,
            "loss": loss,
            "weight_checkpoint": weight_checkpoint,
            "state_checkpoint": state_checkpoint,
            "metrics": metrics or {},
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        topk = [item for item in topk if item.get("weight_checkpoint") != weight_checkpoint]
        topk.append(candidate)

    topk = [item for item in topk if _existing_file(item.get("weight_checkpoint")) and _finite_number(item.get("loss"))]
    topk.sort(key=lambda item: (float(item["loss"]), -int(item.get("steps", 0))))
    keep = topk[: max(k, 0)]
    dropped = topk[max(k, 0) :]

    keep_weights = {item.get("weight_checkpoint") for item in keep}
    keep_states = {str(Path(item["state_checkpoint"]).resolve()) for item in keep if item.get("state_checkpoint")}
    if keep_latest_state:
        keep_states.add(str(Path(keep_latest_state).resolve()))

    for item in dropped:
        _remove_file(item.get("weight_checkpoint"))
        state = item.get("state_checkpoint")
        if state and str(Path(state).resolve()) not in keep_states:
            _remove_dir(state)

    if weight_checkpoint:
        _prune_weight_dir(Path(weight_checkpoint).parent, keep_weights)
    if state_checkpoint:
        _prune_state_dir(Path(state_checkpoint).parent, keep_states)

    index = {
        "metric": "loss",
        "mode": "min",
        "keep_top_k": k,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoints": keep,
    }
    index_path.write_text(_json_dumps(index), encoding="utf-8")

    current_rank = None
    if weight_checkpoint:
        for rank, item in enumerate(keep, start=1):
            if item.get("weight_checkpoint") == weight_checkpoint:
                current_rank = rank
                break
        if current_rank is None:
            _remove_file(weight_checkpoint)
            if state_checkpoint and str(Path(state_checkpoint).resolve()) not in keep_states:
                _remove_dir(state_checkpoint)

    index["current_loss"] = loss
    index["current_topk_rank"] = current_rank
    return index


def _load_topk(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    checkpoints = data.get("checkpoints") if isinstance(data, dict) else data
    return checkpoints if isinstance(checkpoints, list) else []


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _existing_file(path: Any) -> bool:
    return bool(path) and Path(path).is_file()


def _remove_file(path: Any) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _remove_dir(path: Any) -> None:
    if not path:
        return
    target = Path(path)
    if target.name == "latest":
        return
    try:
        if target.exists():
            shutil.rmtree(target)
    except OSError:
        pass


def _prune_weight_dir(checkpoint_dir: Path, keep_weights: set[Any]) -> None:
    keep = {str(Path(path).resolve()) for path in keep_weights if path}
    for pattern in ["steps_*_pytorch_model.pt", "steps_*_model.safetensors"]:
        for path in checkpoint_dir.glob(pattern):
            if str(path.resolve()) not in keep:
                _remove_file(path)


def _prune_state_dir(state_root: Path, keep_states: set[str]) -> None:
    for path in state_root.glob("steps_*"):
        if not path.is_dir():
            continue
        if str(path.resolve()) not in keep_states:
            _remove_dir(path)
