from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


ULTIMATE_CALVIN_MOE_CKPTS = [
    "/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/GTY/runs/abc_augmented_moe_GTY_0519_092147/checkpoints/steps_60000_pytorch_model.pt",
    "/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/GTY/runs/gty_moe_posttrain_8h_GTY_0519_182014/checkpoints/steps_95000_pytorch_model.pt",
    "/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_augmented_moe_adaptive_WMH_bs64_s15k_0519_121249/checkpoints/steps_15000_pytorch_model.pt",
    "/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_augmented_moe_adaptive_WMH_bs64_s15k_0519_121249/final_model/pytorch_model.pt",
]

ULTIMATE_CALVIN_MOE_WEIGHTS = [0.38, 0.42, 0.10, 0.10]


@dataclass(frozen=True)
class EnsembleMember:
    path: Path
    weight: float
    run_dir: Path
    config_path: Path
    stats_path: Path
    num_keys: int
    file_size: int
    sample_digest: str


def recipe_checkpoints(recipe: str | None) -> tuple[list[str], list[float] | None]:
    if not recipe:
        return [], None
    if recipe == "calvin_ultimate_moe":
        return list(ULTIMATE_CALVIN_MOE_CKPTS), list(ULTIMATE_CALVIN_MOE_WEIGHTS)
    raise ValueError(f"Unknown ensemble recipe: {recipe}")


def parse_weights(raw: str | None, n: int, defaults: list[float] | None = None) -> list[float]:
    if raw and raw.strip():
        weights = [float(part) for part in raw.replace(",", " ").split() if part.strip()]
    elif defaults is not None:
        weights = list(defaults)
    else:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"Expected {n} ensemble weights, got {len(weights)}")
    if any(weight < 0 for weight in weights):
        raise ValueError("Ensemble weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    return [weight / total for weight in weights]


def checkpoint_run_dir(path: Path) -> Path:
    if path.parent.name in {"checkpoints", "final_model"}:
        return path.parents[1]
    return path.parent


def sample_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    offsets = [0]
    if size > chunk_size * 3:
        offsets.append(max(0, size // 2 - chunk_size // 2))
    if size > chunk_size:
        offsets.append(max(0, size - chunk_size))
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        for offset in dict.fromkeys(offsets):
            f.seek(offset)
            h.update(f.read(chunk_size))
    return h.hexdigest()


def _torch_load(path: Path) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if path.suffix == ".pt":
        kwargs["mmap"] = os.getenv("STARVLA_TORCH_LOAD_MMAP", "1").strip().lower() not in {"0", "false", "no", "off"}
        kwargs["weights_only"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)
    except Exception:
        kwargs.pop("mmap", None)
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def inspect_members(paths: list[str], weights: list[float]) -> list[EnsembleMember]:
    members: list[EnsembleMember] = []
    for raw, weight in zip(paths, weights):
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        run_dir = checkpoint_run_dir(path)
        config_path = run_dir / "config.yaml"
        stats_path = run_dir / "dataset_statistics.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.yaml for checkpoint: {path}")
        if not stats_path.is_file():
            raise FileNotFoundError(f"Missing dataset_statistics.json for checkpoint: {path}")
        state = _torch_load(path)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint is not a state_dict: {path}")
        members.append(
            EnsembleMember(
                path=path,
                weight=float(weight),
                run_dir=run_dir,
                config_path=config_path,
                stats_path=stats_path,
                num_keys=len(state),
                file_size=path.stat().st_size,
                sample_digest=sample_digest(path),
            )
        )
        del state
        gc.collect()
    return members


def _validate_compatible(states: list[dict[str, torch.Tensor]], members: list[EnsembleMember], *, strict: bool) -> list[str]:
    ref_keys = list(states[0].keys())
    ref_key_set = set(ref_keys)
    common = set(ref_keys)
    problems: list[str] = []
    for state, member in zip(states[1:], members[1:]):
        keys = set(state.keys())
        missing = sorted(ref_key_set - keys)
        extra = sorted(keys - ref_key_set)
        if missing or extra:
            message = f"{member.path}: missing={len(missing)} extra={len(extra)}"
            if strict:
                raise RuntimeError(message)
            problems.append(message)
        common &= keys
    ordered = [key for key in ref_keys if key in common]
    if not ordered:
        raise RuntimeError("No common keys found across ensemble checkpoints")
    for key in ordered:
        ref = states[0][key]
        for state, member in zip(states[1:], members[1:]):
            value = state[key]
            if tuple(value.shape) != tuple(ref.shape):
                raise RuntimeError(f"Shape mismatch for {key}: {members[0].path}={tuple(ref.shape)} {member.path}={tuple(value.shape)}")
    return ordered


def _write_ensemble_sidecars(
    output_dir: Path,
    output_ckpt: Path,
    members: list[EnsembleMember],
    weights: list[float],
    reference_index: int,
    method: str,
) -> None:
    ref = members[reference_index]
    shutil.copy2(ref.config_path, output_dir / "config.yaml")
    shutil.copy2(ref.stats_path, output_dir / "dataset_statistics.json")
    if (ref.run_dir / "config.full.yaml").is_file():
        shutil.copy2(ref.run_dir / "config.full.yaml", output_dir / "config.full.yaml")

    cfg = OmegaConf.load(output_dir / "config.yaml")
    cfg.output_dir = str(output_dir)
    cfg.run_id = output_dir.name
    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.ensemble = {
        "method": method,
        "output_checkpoint": str(output_ckpt),
        "reference_checkpoint": str(ref.path),
        "members": [str(member.path) for member in members],
        "weights": weights,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OmegaConf.save(cfg, output_dir / "config.yaml")

    metadata = {
        "method": method,
        "output_checkpoint": str(output_ckpt),
        "reference_index": reference_index,
        "reference_checkpoint": str(ref.path),
        "members": [
            {
                "path": str(member.path),
                "run_dir": str(member.run_dir),
                "weight": member.weight,
                "num_keys": member.num_keys,
                "file_size": member.file_size,
                "sample_digest": member.sample_digest,
            }
            for member in members
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "ensemble.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def build_weight_soup(
    *,
    ckpt_paths: list[str],
    weights: list[float],
    output_dir: str | Path,
    reference_index: int = 0,
    strict: bool = True,
    overwrite: bool = False,
    method: str = "weighted_soup",
) -> dict[str, Any]:
    if not ckpt_paths:
        raise ValueError("No checkpoints were provided for ensemble")
    if not (0 <= reference_index < len(ckpt_paths)):
        raise ValueError(f"reference_index must be in [0, {len(ckpt_paths)-1}]")

    members = inspect_members(ckpt_paths, weights)
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_ckpt = checkpoint_dir / "steps_ensemble_pytorch_model.pt"
    if output_ckpt.exists() and not overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_ckpt}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    states = [_torch_load(member.path) for member in members]
    ordered_keys = _validate_compatible(states, members, strict=strict)

    out_state: dict[str, torch.Tensor] = {}
    for index, key in enumerate(ordered_keys, start=1):
        tensors = [state[key] for state in states]
        ref = tensors[0]
        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for tensor, weight in zip(tensors, weights):
                acc.add_(tensor.to(dtype=torch.float32), alpha=float(weight))
            out_state[key] = acc.to(dtype=ref.dtype)
        elif all(torch.equal(ref, tensor) for tensor in tensors[1:]):
            out_state[key] = ref.clone()
        else:
            out_state[key] = ref.clone()
        if index % 100 == 0:
            gc.collect()

    with tempfile.NamedTemporaryFile(prefix=output_ckpt.name, suffix=".tmp", dir=str(checkpoint_dir), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(out_state, tmp_path)
        tmp_path.replace(output_ckpt)
    finally:
        tmp_path.unlink(missing_ok=True)

    _write_ensemble_sidecars(output_dir, output_ckpt, members, weights, reference_index, method)
    del states
    del out_state
    gc.collect()
    return {
        "output_dir": str(output_dir),
        "checkpoint": str(output_ckpt),
        "num_members": len(members),
        "num_keys": len(ordered_keys),
        "weights": weights,
        "reference_checkpoint": str(members[reference_index].path),
        "members": [str(member.path) for member in members],
    }
