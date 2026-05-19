from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import CATALOG_PATH, REPO_ROOT


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def model_path(model_key_or_path: str) -> str:
    catalog = load_catalog()
    models = catalog.get("models", {})
    if model_key_or_path in models:
        return models[model_key_or_path]["path"]
    return model_key_or_path


def slug(value: str | Path | None) -> str:
    text = str(value or "default")
    text = text.strip().replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text[:96] or "default"


def model_name(model_key_or_path: str) -> str:
    catalog = load_catalog()
    if model_key_or_path in catalog.get("models", {}):
        return slug(model_key_or_path)
    return slug(Path(model_key_or_path).name)


def dataset_name(preset_key: str, data_mix: str | None = None, vlm_dataset: str | None = None) -> str:
    catalog = load_catalog()
    preset = catalog.get("train_presets", {}).get(preset_key, {})
    name = data_mix or vlm_dataset or preset.get("data_mix") or preset.get("vlm_dataset") or preset_key
    return slug(name)


def experiment_group_name(model_key_or_path: str, framework: str, dataset: str) -> str:
    return f"{model_name(model_key_or_path)}-{slug(framework)}-{slug(dataset)}"


def default_model_for_framework(framework: str) -> str:
    catalog = load_catalog()
    entry = catalog.get("frameworks", {}).get(framework, {})
    return entry.get("default_model", "qwen35_0_8b")


def label(kind: str, key: str) -> str:
    catalog = load_catalog()
    return catalog.get(kind, {}).get(key, {}).get("label", key)


def keys(kind: str) -> list[str]:
    return list(load_catalog().get(kind, {}).keys())
