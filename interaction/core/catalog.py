from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import CATALOG_PATH, CONFIG_ROOT, REPO_ROOT


POLICY_CATALOG_PATH = CONFIG_ROOT / "policies.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_policy_catalog() -> dict[str, Any]:
    if not POLICY_CATALOG_PATH.exists():
        return {}
    with POLICY_CATALOG_PATH.open("r", encoding="utf-8") as f:
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
    policy_catalog = load_policy_catalog()
    base_models = policy_catalog.get("base_models", {})
    if model_key_or_path in base_models:
        model_key_or_path = base_models[model_key_or_path].get("model", model_key_or_path)
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
    policy_catalog = load_policy_catalog()
    if model_key_or_path in policy_catalog.get("base_models", {}):
        model_key_or_path = policy_catalog["base_models"][model_key_or_path].get("model", model_key_or_path)
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
    policy_catalog = load_policy_catalog()
    if framework in policy_catalog.get("action_experts", {}):
        framework = policy_catalog["action_experts"][framework].get("default_framework", framework)
    entry = catalog.get("frameworks", {}).get(framework, {})
    return entry.get("default_model", "qwen35_0_8b")


def label(kind: str, key: str) -> str:
    catalog = load_catalog()
    return catalog.get(kind, {}).get(key, {}).get("label", key)


def keys(kind: str) -> list[str]:
    return list(load_catalog().get(kind, {}).keys())


def policy_keys(kind: str) -> list[str]:
    return list(load_policy_catalog().get(kind, {}).keys())


def all_model_keys() -> list[str]:
    values = list(keys("models"))
    for key in policy_keys("base_models"):
        if key not in values:
            values.append(key)
    return values


def all_action_expert_keys() -> list[str]:
    values = list(keys("frameworks"))
    for key in policy_keys("action_experts"):
        if key not in values:
            values.append(key)
    return values


def base_model_key(model_key_or_path: str) -> str:
    policy_catalog = load_policy_catalog()
    entry = policy_catalog.get("base_models", {}).get(model_key_or_path)
    if entry:
        return entry.get("model", model_key_or_path)
    return model_key_or_path


def model_family(model_key_or_path: str) -> str:
    policy_catalog = load_policy_catalog()
    base_models = policy_catalog.get("base_models", {})
    if model_key_or_path in base_models:
        return str(base_models[model_key_or_path].get("family", "qwen"))
    resolved_key = base_model_key(model_key_or_path)
    text = f"{model_key_or_path} {resolved_key} {model_path(resolved_key)}".lower()
    if "cosmos" in text or "cosmo" in text:
        return "cosmos"
    if "wan" in text:
        return "wan"
    if "gemma" in text:
        return "gemma"
    if "intern" in text:
        return "intern"
    return "qwen"


def resolve_action_expert(action_expert: str, model_key_or_path: str) -> tuple[str, dict[str, Any]]:
    policy_catalog = load_policy_catalog()
    entry = dict(policy_catalog.get("action_experts", {}).get(action_expert, {}))
    if not entry:
        return action_expert, {"label": label("frameworks", action_expert), "default_framework": action_expert}
    family = model_family(model_key_or_path)
    framework = entry.get("framework_by_family", {}).get(family) or entry.get("default_framework") or action_expert
    entry["resolved_family"] = family
    entry["resolved_framework"] = framework
    return framework, entry


def policy_combo_slug(training_policies: list[str] | None, structure_policies: list[str] | None) -> str:
    parts: list[str] = []
    if training_policies:
        parts.append("+".join(slug(item) for item in training_policies))
    if structure_policies:
        parts.append("+".join(slug(item) for item in structure_policies))
    return slug("__".join(parts)) if parts else ""
