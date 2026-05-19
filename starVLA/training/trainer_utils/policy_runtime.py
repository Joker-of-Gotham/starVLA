"""Runtime execution hooks for StarVLA training/structure policies.

The interaction layer passes policy IDs into ``trainer.policy_runtime``.  This
module makes those IDs executable inside the native trainers instead of leaving
them as launch metadata.  Whenever a framework exposes a dedicated loss key, the
runtime uses it.  Otherwise it falls back to cheap, bounded regularizers and
batch weighting that keep the run valid for post-training from an existing
checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw: Iterable[Any] = value
    else:
        raw = str(value).replace(",", " ").split()
    return [str(item).strip() for item in raw if str(item).strip()]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


@dataclass
class PolicyRuntime:
    enabled: bool
    training_policies: list[str]
    structure_policies: list[str]
    posttrain: bool = False
    action_expert_key: str = ""
    resolved_framework: str = ""
    advantage_weighting: bool = False
    horizon_curriculum: bool = False
    safety_projection: bool = False
    checkpoint_averaging: bool = False
    representation_alignment_weight: float = 0.0
    future_world_weight: float = 0.0
    intent_bottleneck_weight: float = 0.0
    video_retarget_weight: float = 0.0
    uncertainty_weight: float = 0.0
    anti_forgetting_weight: float = 0.0
    domain_adapter_weight: float = 0.0
    rl_regularization_weight: float = 0.0
    hybrid_residual_weight: float = 0.0
    grouped_action_weight: float = 0.0
    moe_regularization_weight: float = 0.0
    action_token_warmup_weight: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "PolicyRuntime":
        runtime_cfg = _cfg_get(_cfg_get(cfg, "trainer", None), "policy_runtime", None)
        training = _split_ids(_cfg_get(runtime_cfg, "training_policies", ""))
        structure = _split_ids(_cfg_get(runtime_cfg, "structure_policies", ""))
        enabled = _as_bool(_cfg_get(runtime_cfg, "enabled", bool(training or structure)), bool(training or structure))
        return cls(
            enabled=enabled,
            training_policies=training,
            structure_policies=structure,
            posttrain=_as_bool(_cfg_get(runtime_cfg, "posttrain", False)),
            action_expert_key=str(_cfg_get(runtime_cfg, "action_expert_key", "") or ""),
            resolved_framework=str(_cfg_get(runtime_cfg, "resolved_framework", "") or ""),
            advantage_weighting=_as_bool(_cfg_get(runtime_cfg, "advantage_weighting", False)),
            horizon_curriculum=_as_bool(_cfg_get(runtime_cfg, "horizon_curriculum", False)),
            safety_projection=_as_bool(_cfg_get(runtime_cfg, "safety_projection", False)),
            checkpoint_averaging=_as_bool(_cfg_get(runtime_cfg, "checkpoint_averaging", False)),
            representation_alignment_weight=_as_float(_cfg_get(runtime_cfg, "representation_alignment_weight", 0.0)),
            future_world_weight=_as_float(_cfg_get(runtime_cfg, "future_world_weight", 0.0)),
            intent_bottleneck_weight=_as_float(_cfg_get(runtime_cfg, "intent_bottleneck_weight", 0.0)),
            video_retarget_weight=_as_float(_cfg_get(runtime_cfg, "video_retarget_weight", 0.0)),
            uncertainty_weight=_as_float(_cfg_get(runtime_cfg, "uncertainty_weight", 0.0)),
            anti_forgetting_weight=_as_float(_cfg_get(runtime_cfg, "anti_forgetting_weight", 0.0)),
            domain_adapter_weight=_as_float(_cfg_get(runtime_cfg, "domain_adapter_weight", 0.0)),
            rl_regularization_weight=_as_float(_cfg_get(runtime_cfg, "rl_regularization_weight", 0.0)),
            hybrid_residual_weight=_as_float(_cfg_get(runtime_cfg, "hybrid_residual_weight", 0.0)),
            grouped_action_weight=_as_float(_cfg_get(runtime_cfg, "grouped_action_weight", 0.0)),
            moe_regularization_weight=_as_float(_cfg_get(runtime_cfg, "moe_regularization_weight", 0.0)),
            action_token_warmup_weight=_as_float(_cfg_get(runtime_cfg, "action_token_warmup_weight", 0.0)),
        )

    @property
    def ids(self) -> set[str]:
        return set(self.training_policies) | set(self.structure_policies)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "training_policies": self.training_policies,
            "structure_policies": self.structure_policies,
            "posttrain": self.posttrain,
            "action_expert_key": self.action_expert_key,
            "resolved_framework": self.resolved_framework,
        }

    def apply_action_loss(
        self,
        *,
        action_loss: torch.Tensor,
        output_dict: dict[str, Any],
        model: torch.nn.Module,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.enabled:
            return action_loss, {}

        total_loss = action_loss
        metrics: dict[str, float] = {}

        if self.advantage_weighting:
            weight = self._batch_advantage_weight(batch, action_loss.device, action_loss.dtype)
            total_loss = total_loss * weight
            metrics["policy/advantage_weight"] = float(weight.detach().float().item())

        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("representation_loss", "alignment_loss", "repa_loss", "future_repr_loss"),
            fallback_names=("qwen_vl_interface", "action_model"),
            weight=self.representation_alignment_weight,
            metric="policy/representation_alignment",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("future_loss", "world_loss", "wam_loss"),
            fallback_names=("action_model", "world_model"),
            weight=self.future_world_weight,
            metric="policy/future_world",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("intent_loss", "latent_intent_loss"),
            fallback_names=("action_model",),
            weight=self.intent_bottleneck_weight,
            metric="policy/intent_bottleneck",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("video_retarget_loss", "inverse_dynamics_loss"),
            fallback_names=("action_model",),
            weight=self.video_retarget_weight,
            metric="policy/video_retarget",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("uncertainty_loss", "nll_loss", "mixture_loss"),
            fallback_names=("action_model",),
            weight=self.uncertainty_weight,
            metric="policy/uncertainty",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("anti_forgetting_loss", "distill_loss", "kl_loss"),
            fallback_names=("qwen_vl_interface", "action_model"),
            weight=self.anti_forgetting_weight,
            metric="policy/anti_forgetting",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("domain_loss", "mmd_loss", "adapter_loss"),
            fallback_names=("qwen_vl_interface", "action_model"),
            weight=self.domain_adapter_weight,
            metric="policy/domain_adapter",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("rl_loss", "value_loss", "reward_loss", "preference_loss", "dpo_loss"),
            fallback_names=("action_model",),
            weight=self.rl_regularization_weight,
            metric="policy/rl_regularization",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("hybrid_residual_loss", "token_loss", "residual_loss"),
            fallback_names=("action_model",),
            weight=self.hybrid_residual_weight + self.action_token_warmup_weight * 0.01,
            metric="policy/hybrid_token",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("grouped_action_loss", "gripper_loss", "contact_loss", "bimanual_loss"),
            fallback_names=("action_model",),
            weight=self.grouped_action_weight,
            metric="policy/grouped_action",
        )
        total_loss = self._add_output_or_regularizer(
            total_loss, output_dict, model, metrics,
            keys=("moe_balance_loss", "router_loss"),
            fallback_names=("action_model",),
            weight=self.moe_regularization_weight,
            metric="policy/moe",
        )

        metrics["policy/enabled"] = 1.0
        metrics["policy/posttrain"] = 1.0 if self.posttrain else 0.0
        return total_loss, metrics

    def apply_vlm_loss(
        self,
        *,
        vlm_loss: torch.Tensor,
        model: torch.nn.Module,
        batch: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.enabled:
            return vlm_loss, {}
        metrics: dict[str, float] = {"policy/enabled": 1.0, "policy/posttrain": 1.0 if self.posttrain else 0.0}
        total_loss = vlm_loss
        if self.representation_alignment_weight or self.domain_adapter_weight or self.anti_forgetting_weight:
            weight = self.representation_alignment_weight + self.domain_adapter_weight + self.anti_forgetting_weight
            reg = self._param_regularizer(model, ("qwen_vl_interface",), vlm_loss.device, vlm_loss.dtype)
            if reg is not None:
                total_loss = total_loss + weight * reg
                metrics["policy/vlm_regularizer"] = float(reg.detach().float().item())
        return total_loss, metrics

    def _add_output_or_regularizer(
        self,
        loss: torch.Tensor,
        output_dict: dict[str, Any],
        model: torch.nn.Module,
        metrics: dict[str, float],
        *,
        keys: tuple[str, ...],
        fallback_names: tuple[str, ...],
        weight: float,
        metric: str,
    ) -> torch.Tensor:
        if weight <= 0:
            return loss
        value = None
        for key in keys:
            candidate = output_dict.get(key)
            if isinstance(candidate, torch.Tensor):
                value = candidate.mean()
                break
        if value is None:
            value = self._param_regularizer(model, fallback_names, loss.device, loss.dtype)
        if value is None:
            return loss
        metrics[metric] = float(value.detach().float().item())
        return loss + float(weight) * value

    def _param_regularizer(
        self,
        model: torch.nn.Module,
        name_fragments: tuple[str, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        total = torch.zeros((), device=device, dtype=dtype)
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name_fragments and not any(fragment in name for fragment in name_fragments):
                continue
            total = total + param.float().pow(2).mean().to(device=device, dtype=dtype)
            count += 1
            if count >= 24:
                break
        if count == 0:
            return None
        return total / count

    def _batch_advantage_weight(self, batch: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values: list[float] = []
        examples = batch if isinstance(batch, list) else []
        for example in examples:
            if not isinstance(example, dict):
                continue
            for key in ("advantage", "weight", "reward", "return", "success"):
                if key in example:
                    try:
                        values.append(float(example[key]))
                    except (TypeError, ValueError):
                        pass
                    break
            if "failure" in example:
                try:
                    values.append(1.0 + float(example["failure"]))
                except (TypeError, ValueError):
                    pass
        if not values:
            return torch.ones((), device=device, dtype=dtype)
        mean_value = sum(values) / len(values)
        weight = max(0.1, min(10.0, 1.0 + mean_value))
        return torch.tensor(weight, device=device, dtype=dtype)
