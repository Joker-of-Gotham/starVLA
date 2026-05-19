from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERACTION_ROOT = REPO_ROOT / "interaction"
CONFIG_ROOT = INTERACTION_ROOT / "config"
DATA_STARVLA_ROOT = Path(
    os.getenv("STARVLA_DATA_ROOT", "/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla")
)
DEFAULT_CHECKPOINT_ROOT = Path(os.getenv("STARVLA_CHECKPOINT_ROOT", str(DATA_STARVLA_ROOT / "checkpoints")))
DESIRED_RUNS_ROOT = Path(os.getenv("STARVLA_RUNS_ROOT", str(DATA_STARVLA_ROOT / "runs")))
FALLBACK_RUNS_ROOT = INTERACTION_ROOT / "runs"
RUNS_ROOT = DESIRED_RUNS_ROOT
ACTIVATE_SCRIPT = REPO_ROOT / "bar" / "activate_starvla_no_proxy.sh"
DEFAULT_PYTHON = REPO_ROOT / "env" / "starvla-py310" / "bin" / "python"
DEFAULT_ACCELERATE_CONFIG = REPO_ROOT / "starVLA" / "config" / "deepseeds" / "deepspeed_zero2.yaml"
CATALOG_PATH = CONFIG_ROOT / "presets.json"


def _path_writable(path: Path, create: bool = False) -> bool:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.exists() or not path.is_dir():
            return False
        probe = path / ".starvla_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def runtime_runs_root() -> Path:
    """Return the active interaction run directory.

    The production target is DESIRED_RUNS_ROOT. In this Codex sandbox that
    target may be read-only, so local self-tests fall back to interaction/runs.
    """
    if _path_writable(DESIRED_RUNS_ROOT, create=True):
        return DESIRED_RUNS_ROOT
    FALLBACK_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return FALLBACK_RUNS_ROOT


def candidate_runs_roots() -> list[Path]:
    roots: list[Path] = []
    for root in [runtime_runs_root(), DESIRED_RUNS_ROOT, FALLBACK_RUNS_ROOT]:
        if root not in roots and root.exists():
            roots.append(root)
    return roots


def artifact_roots_report() -> dict[str, Any]:
    active = runtime_runs_root()
    return {
        "data_root": str(DATA_STARVLA_ROOT),
        "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "checkpoint_root_exists": DEFAULT_CHECKPOINT_ROOT.exists(),
        "checkpoint_root_writable": _path_writable(DEFAULT_CHECKPOINT_ROOT),
        "desired_runs_root": str(DESIRED_RUNS_ROOT),
        "desired_runs_root_exists": DESIRED_RUNS_ROOT.exists(),
        "desired_runs_root_writable": _path_writable(DESIRED_RUNS_ROOT),
        "active_runs_root": str(active),
        "fallback_runs_root": str(FALLBACK_RUNS_ROOT),
        "using_fallback_runs_root": active != DESIRED_RUNS_ROOT,
    }


def ensure_runtime_dirs() -> Path:
    return runtime_runs_root()


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
