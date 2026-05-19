from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .gpu import detect_gpus


@dataclass(frozen=True)
class ThroughputSettings:
    requested_profile: str
    resolved_profile: str
    vla_batch_size: int | None
    vlm_batch_size: int | None
    dataloader_workers: int | None
    prefetch_factor: int | None
    pin_memory: bool | None
    persistent_workers: bool | None
    env_defaults: dict[str, str]
    warnings: list[str]
    notes: list[str]


def resolve_throughput_settings(
    *,
    profile: str,
    selected_gpus: list[int],
    model_key: str,
    framework: str,
    mode: str,
    preset_vla_batch: int | None,
    preset_vlm_batch: int | None,
    explicit_vla_batch: int | None,
    explicit_vlm_batch: int | None,
    explicit_workers: int | None,
    explicit_prefetch: int | None,
    explicit_pin_memory: bool | None,
    explicit_persistent_workers: bool | None,
) -> ThroughputSettings:
    resolved = _resolve_profile(profile, selected_gpus)
    notes: list[str] = []
    warnings: list[str] = []
    env_defaults: dict[str, str] = {}

    if resolved == "none":
        vla_batch = explicit_vla_batch if explicit_vla_batch is not None else preset_vla_batch
        vlm_batch = explicit_vlm_batch if explicit_vlm_batch is not None else preset_vlm_batch
        return ThroughputSettings(
            requested_profile=profile,
            resolved_profile=resolved,
            vla_batch_size=vla_batch,
            vlm_batch_size=vlm_batch,
            dataloader_workers=explicit_workers,
            prefetch_factor=explicit_prefetch,
            pin_memory=explicit_pin_memory,
            persistent_workers=explicit_persistent_workers,
            env_defaults=env_defaults,
            warnings=warnings,
            notes=["Throughput profile disabled; using preset/config loader defaults."],
        )

    model_target = _batch_target_for_model(model_key, framework, mode, resolved)
    vla_batch = _choose_batch(
        explicit=explicit_vla_batch,
        preset=preset_vla_batch,
        target=model_target.get("vla"),
        model_key=model_key,
    )
    vlm_batch = _choose_batch(
        explicit=explicit_vlm_batch,
        preset=preset_vlm_batch,
        target=model_target.get("vlm"),
        model_key=model_key,
    )
    safe_cap = _explicit_batch_cap(model_key, framework, mode, resolved)
    if safe_cap:
        cap_vla = safe_cap.get("vla")
        cap_vlm = safe_cap.get("vlm")
        if explicit_vla_batch is not None and cap_vla is not None and vla_batch is not None and vla_batch > cap_vla:
            warnings.append(
                f"VLA per-device batch {vla_batch} is unsafe for {framework}/{model_key} under {resolved}; "
                f"clamped to {cap_vla}. Use throughput profile 'none' only for deliberate profiling."
            )
            vla_batch = cap_vla
        if explicit_vlm_batch is not None and cap_vlm is not None and vlm_batch is not None and vlm_batch > cap_vlm:
            warnings.append(
                f"VLM per-device batch {vlm_batch} is unsafe for {framework}/{model_key} under {resolved}; "
                f"clamped to {cap_vlm}. Use throughput profile 'none' only for deliberate profiling."
            )
            vlm_batch = cap_vlm

    if resolved == "h200_saturated":
        workers = 8
        prefetch = 4
        env_defaults.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                "NVIDIA_TF32_OVERRIDE": "1",
            }
        )
        notes.append("H200 throughput profile raises per-device batch where safe and enables persistent prefetched workers.")
    elif resolved == "balanced":
        workers = 4
        prefetch = 2
        env_defaults.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
            }
        )
        notes.append("Balanced throughput profile keeps moderate dataloader parallelism and TF32-friendly defaults.")
    else:
        workers = 2
        prefetch = 2
        env_defaults.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        notes.append("Conservative throughput profile avoids aggressive worker and batch scaling.")

    if explicit_workers is not None:
        workers = explicit_workers
    if explicit_prefetch is not None:
        prefetch = explicit_prefetch
    pin_memory = explicit_pin_memory if explicit_pin_memory is not None else True
    persistent_workers = explicit_persistent_workers if explicit_persistent_workers is not None else workers > 0

    if selected_gpus and len(selected_gpus) >= 8 and resolved != "h200_saturated":
        warnings.append("8 GPUs selected but h200_saturated was not used; pass --throughput-profile h200_saturated to force the aggressive profile.")
    if explicit_vla_batch is None and preset_vla_batch is not None and vla_batch is not None and vla_batch > preset_vla_batch:
        notes.append(f"VLA per-device batch raised from preset {preset_vla_batch} to {vla_batch}.")
    if explicit_vlm_batch is None and preset_vlm_batch is not None and vlm_batch is not None and vlm_batch > preset_vlm_batch:
        notes.append(f"VLM per-device batch raised from preset {preset_vlm_batch} to {vlm_batch}.")

    return ThroughputSettings(
        requested_profile=profile,
        resolved_profile=resolved,
        vla_batch_size=vla_batch,
        vlm_batch_size=vlm_batch,
        dataloader_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        env_defaults=env_defaults,
        warnings=warnings,
        notes=notes,
    )


def _resolve_profile(profile: str, selected_gpus: list[int]) -> str:
    if profile != "auto":
        return profile
    visible = {gpu.index: gpu for gpu in detect_gpus()}
    selected = [visible[index] for index in selected_gpus if index in visible]
    if len(selected_gpus) >= 8 and selected and all(gpu.is_h200 for gpu in selected):
        return "h200_saturated"
    if len(selected_gpus) >= 4:
        return "balanced"
    return "conservative"


def _batch_target_for_model(model_key: str, framework: str, mode: str, profile: str) -> dict[str, int]:
    key = str(model_key).lower()
    fw = str(framework).lower()
    is_cotrain = mode == "cotrain"
    is_fast = fw.endswith("fast") or "fast" in fw

    if profile == "conservative":
        if is_fast:
            return {"vla": 2, "vlm": 1}
        if "cosmos" in key or "cosmo" in fw:
            return {"vla": 2, "vlm": 1}
        return {"vla": 2 if "9b" in key else 4, "vlm": 1 if "9b" in key else 2}
    if profile == "balanced":
        if is_fast:
            return {"vla": 4, "vlm": 1}
        if "cosmos" in key or "cosmo" in fw:
            return {"vla": 4, "vlm": 1}
        if "9b" in key:
            return {"vla": 4, "vlm": 1}
        if "4b" in key or "qwen3-vl" in key or "qwen3vl" in key:
            return {"vla": 8, "vlm": 2}
        return {"vla": 8, "vlm": 2}

    # h200_saturated: use H200 memory aggressively while keeping large VLMs guarded.
    if is_fast:
        return {"vla": 8, "vlm": 2}
    if "cosmos" in key or "cosmo" in fw:
        return {"vla": 8, "vlm": 1}
    if "9b" in key:
        return {"vla": 4, "vlm": 1}
    if "4b" in key or "qwen3-vl" in key or "qwen3vl" in key:
        return {"vla": 8 if is_cotrain else 12, "vlm": 2}
    if "2b" in key:
        return {"vla": 16 if is_cotrain else 24, "vlm": 4}
    if "0.8b" in key or "0_8b" in key:
        return {"vla": 24 if is_cotrain else 32, "vlm": 4}
    return {"vla": 8, "vlm": 2}


def _explicit_batch_cap(model_key: str, framework: str, mode: str, profile: str) -> dict[str, int] | None:
    if profile == "none":
        return None
    fw = str(framework).lower()
    key = str(model_key).lower()
    is_fast = fw.endswith("fast") or "fast" in fw
    if not is_fast:
        return None
    # FAST/action-token training goes through the VLM sequence path and is much
    # more memory sensitive than continuous OFT/PI heads. Keep manual input from
    # bypassing the profile guard; the unsafe override remains profile=none.
    if profile == "h200_saturated" and ("4b" in key or "qwen3-vl" in key or "qwen3vl" in key):
        return {"vla": 16, "vlm": 2}
    if profile == "h200_saturated":
        return {"vla": 24, "vlm": 2}
    if profile == "balanced":
        return {"vla": 8, "vlm": 1}
    return {"vla": 4, "vlm": 1}


def _choose_batch(*, explicit: int | None, preset: int | None, target: int | None, model_key: str) -> int | None:
    if explicit is not None:
        return explicit
    if target is None:
        return preset
    if preset is None:
        return target
    key = str(model_key).lower()
    if "9b" in key:
        return min(preset, target)
    return max(preset, target)


def as_meta(settings: ThroughputSettings) -> dict[str, Any]:
    return {
        "throughput_profile": settings.resolved_profile,
        "throughput_profile_requested": settings.requested_profile,
        "vla_batch_size": settings.vla_batch_size,
        "vlm_batch_size": settings.vlm_batch_size,
        "dataloader_workers": settings.dataloader_workers,
        "dataloader_prefetch_factor": settings.prefetch_factor,
        "dataloader_pin_memory": settings.pin_memory,
        "dataloader_persistent_workers": settings.persistent_workers,
        "throughput_env_defaults": settings.env_defaults,
        "throughput_warnings": settings.warnings,
        "throughput_notes": settings.notes,
    }
