from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


_MISSING = object()


def _cfg_get(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        value = _MISSING
        try:
            if hasattr(cur, "get"):
                value = cur.get(part, _MISSING)
        except Exception:
            value = _MISSING
        if value is _MISSING:
            try:
                value = getattr(cur, part)
            except Exception:
                return default
        cur = value
    return cur


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


@dataclass
class AdaptiveLrDecision:
    state: str
    reason: str
    adjusted: bool = False


class AdaptiveLrController:
    """Conservative EMA-based LR multiplier on top of the configured scheduler.

    This controller does not replace warmup/cosine/one-cycle schedules. It keeps
    per-group multiplicative scales and reapplies them after each scheduler step.
    Decisions are made only when the trainer reports metrics, not per noisy batch.
    """

    def __init__(self, optimizer, scheduler, cfg, logger=None):
        adaptive_cfg = _cfg_get(cfg, "trainer.adaptive_lr", {})
        self.enabled = _as_bool(_cfg_get(adaptive_cfg, "enabled", None), True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        self.metric = str(_cfg_get(adaptive_cfg, "metric", "action_dit_loss"))
        self.ema_beta = _as_float(_cfg_get(adaptive_cfg, "ema_beta", 0.98), 0.98)
        self.threshold = _as_float(_cfg_get(adaptive_cfg, "threshold", 1e-3), 1e-3)
        self.explode_factor = _as_float(_cfg_get(adaptive_cfg, "explode_factor", 1.5), 1.5)
        self.reduce_factor = _as_float(_cfg_get(adaptive_cfg, "reduce_factor", 0.5), 0.5)
        self.emergency_reduce_factor = _as_float(
            _cfg_get(adaptive_cfg, "emergency_reduce_factor", 0.25),
            0.25,
        )
        self.increase_factor = _as_float(_cfg_get(adaptive_cfg, "increase_factor", 1.1), 1.1)
        self.plateau_patience = _as_int(_cfg_get(adaptive_cfg, "plateau_patience", 4), 4)
        self.rising_patience = _as_int(_cfg_get(adaptive_cfg, "rising_patience", 3), 3)
        self.cooldown_windows_default = _as_int(_cfg_get(adaptive_cfg, "cooldown_windows", 2), 2)
        self.min_lr = _as_float(_cfg_get(adaptive_cfg, "min_lr", 1e-7), 1e-7)
        self.max_lr_scale = _as_float(_cfg_get(adaptive_cfg, "max_lr_scale", 1.0), 1.0)
        self.low_update_ratio = _as_float(_cfg_get(adaptive_cfg, "low_update_ratio", 1e-4), 1e-4)
        self.rollback_bad_factor = _as_float(_cfg_get(adaptive_cfg, "rollback_bad_factor", 1.03), 1.03)
        self.grad_norm_explode_factor = _as_float(
            _cfg_get(adaptive_cfg, "grad_norm_explode_factor", 4.0),
            4.0,
        )
        self.final_no_increase_fraction = _as_float(
            _cfg_get(adaptive_cfg, "final_no_increase_fraction", 0.8),
            0.8,
        )
        self.min_increase_step = _as_int(
            _cfg_get(adaptive_cfg, "min_increase_step", _cfg_get(cfg, "trainer.num_warmup_steps", 0)),
            0,
        )
        self.total_steps = _as_int(_cfg_get(cfg, "trainer.max_train_steps", 0), 0)

        self.base_lrs = list(getattr(scheduler, "base_lrs", [])) or [
            float(group.get("lr", 0.0)) for group in optimizer.param_groups
        ]
        self.lr_scales = [1.0 for _ in optimizer.param_groups]
        self.loss_ema: float | None = None
        self.best_loss_ema: float | None = None
        self.prev_loss_ema: float | None = None
        self.grad_norm_ema: float | None = None
        self.no_improve_windows = 0
        self.rising_windows = 0
        self.cooldown_windows = 0
        self.windows_since_reduce = 10**9
        self.pending_probe: dict[str, Any] | None = None
        self.last_decision = AdaptiveLrDecision("init", "adaptive_lr_initialized", False)

    def apply_current_schedule(self) -> None:
        if not self.enabled:
            return
        try:
            scheduled_lrs = list(self.scheduler.get_last_lr())
        except Exception:
            scheduled_lrs = [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]
        for idx, group in enumerate(self.optimizer.param_groups):
            scheduled = float(scheduled_lrs[min(idx, len(scheduled_lrs) - 1)])
            base = float(self.base_lrs[min(idx, len(self.base_lrs) - 1)])
            max_lr = max(base * self.max_lr_scale, self.min_lr)
            group["lr"] = min(max(scheduled * self.lr_scales[idx], self.min_lr), max_lr)

    def update(self, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"adaptive_lr/enabled": False}

        metric_value = self._metric_value(metrics)
        grad_norm = metrics.get("grad_norm")
        update_ratio = metrics.get("update_ratio")
        decision = AdaptiveLrDecision("hold", "metric_window", False)

        if _finite(grad_norm):
            grad_norm = float(grad_norm)
            self.grad_norm_ema = (
                grad_norm
                if self.grad_norm_ema is None
                else self.ema_beta * self.grad_norm_ema + (1.0 - self.ema_beta) * grad_norm
            )

        if not _finite(metric_value):
            decision = self._reduce("explode", "nonfinite_metric", self.emergency_reduce_factor)
            self._enter_cooldown()
            self.last_decision = decision
            self.apply_current_schedule()
            return self._metrics(decision, metric_value, update_ratio)

        loss = float(metric_value)
        self.prev_loss_ema = self.loss_ema
        self.loss_ema = loss if self.loss_ema is None else self.ema_beta * self.loss_ema + (1.0 - self.ema_beta) * loss

        if self.best_loss_ema is None or self.loss_ema < self.best_loss_ema * (1.0 - self.threshold):
            self.best_loss_ema = self.loss_ema
            self.no_improve_windows = 0
        else:
            self.no_improve_windows += 1

        if self.prev_loss_ema is not None and self.loss_ema > self.prev_loss_ema * (1.0 + self.threshold):
            self.rising_windows += 1
        else:
            self.rising_windows = 0

        if self.pending_probe and self.loss_ema > self.pending_probe["loss_ema"] * self.rollback_bad_factor:
            self.lr_scales = list(self.pending_probe["lr_scales"])
            self.pending_probe = None
            self._enter_cooldown()
            decision = AdaptiveLrDecision("rollback", "probe_worsened", True)
        elif self._is_exploding(grad_norm):
            decision = self._reduce("explode", "relative_loss_or_grad_worsened", self.emergency_reduce_factor)
            self.pending_probe = None
            self._enter_cooldown()
        elif self.cooldown_windows > 0:
            self.cooldown_windows -= 1
            decision = AdaptiveLrDecision("cooldown", "adjustment_cooldown", False)
        elif self._is_plateau():
            if self._can_probe_up(step, grad_norm, update_ratio):
                self.pending_probe = {"loss_ema": self.loss_ema, "lr_scales": list(self.lr_scales)}
                self._increase()
                self._enter_cooldown()
                decision = AdaptiveLrDecision("probe_up", "plateau_low_update_ratio", True)
            else:
                decision = self._reduce("plateau", "plateau_without_low_update_ratio", self.reduce_factor)
                self._enter_cooldown()

        if decision.adjusted:
            self.no_improve_windows = 0
            self.rising_windows = 0
        self.windows_since_reduce += 1
        self.last_decision = decision
        self.apply_current_schedule()
        return self._metrics(decision, loss, update_ratio)

    def _metric_value(self, metrics: dict[str, Any]) -> Any:
        if self.metric in metrics:
            return metrics[self.metric]
        for key, value in metrics.items():
            lowered = key.lower()
            if key == "loss" or lowered.endswith("_loss") or lowered.endswith("/loss") or "loss" in lowered:
                return value
        return None

    def _is_exploding(self, grad_norm: Any) -> bool:
        if self.best_loss_ema is not None and self.loss_ema is not None:
            if self.loss_ema > self.best_loss_ema * self.explode_factor:
                return True
        if self.rising_windows >= self.rising_patience:
            return True
        if _finite(grad_norm) and self.grad_norm_ema is not None:
            if float(grad_norm) > self.grad_norm_ema * self.grad_norm_explode_factor:
                return True
        return False

    def _is_plateau(self) -> bool:
        return self.no_improve_windows >= self.plateau_patience

    def _can_probe_up(self, step: int, grad_norm: Any, update_ratio: Any) -> bool:
        if self.pending_probe is not None:
            return False
        if step < self.min_increase_step:
            return False
        if self.total_steps and step >= int(self.total_steps * self.final_no_increase_fraction):
            return False
        if self.windows_since_reduce < self.cooldown_windows_default * 2:
            return False
        if not _finite(update_ratio) or float(update_ratio) >= self.low_update_ratio:
            return False
        if _finite(grad_norm) and self.grad_norm_ema is not None:
            if float(grad_norm) > self.grad_norm_ema * self.grad_norm_explode_factor:
                return False
        return any(self._actual_lr(idx) < self._max_lr(idx) * (1.0 - 1e-6) for idx in range(len(self.lr_scales)))

    def _reduce(self, state: str, reason: str, factor: float) -> AdaptiveLrDecision:
        factor = min(max(float(factor), 1e-3), 1.0)
        self.lr_scales = [max(scale * factor, self.min_lr / max(self._base_lr(idx), self.min_lr)) for idx, scale in enumerate(self.lr_scales)]
        self.windows_since_reduce = 0
        return AdaptiveLrDecision(state, reason, True)

    def _increase(self) -> None:
        factor = min(max(float(self.increase_factor), 1.0), 1.2)
        for idx, scale in enumerate(self.lr_scales):
            scheduled = self._scheduled_lr(idx)
            if scheduled <= 0:
                continue
            max_scale = self._max_lr(idx) / scheduled
            self.lr_scales[idx] = min(scale * factor, max_scale)

    def _enter_cooldown(self) -> None:
        self.cooldown_windows = max(self.cooldown_windows, self.cooldown_windows_default)

    def _scheduled_lr(self, idx: int) -> float:
        try:
            scheduled = list(self.scheduler.get_last_lr())
            return float(scheduled[min(idx, len(scheduled) - 1)])
        except Exception:
            return float(self.optimizer.param_groups[idx].get("lr", 0.0))

    def _base_lr(self, idx: int) -> float:
        return float(self.base_lrs[min(idx, len(self.base_lrs) - 1)])

    def _max_lr(self, idx: int) -> float:
        return max(self._base_lr(idx) * self.max_lr_scale, self.min_lr)

    def _actual_lr(self, idx: int) -> float:
        return float(self.optimizer.param_groups[idx].get("lr", 0.0))

    def _metrics(self, decision: AdaptiveLrDecision, loss_value: Any, update_ratio: Any) -> dict[str, Any]:
        out = {
            "adaptive_lr/enabled": True,
            "adaptive_lr/state": decision.state,
            "adaptive_lr/reason": decision.reason,
            "adaptive_lr/adjusted": decision.adjusted,
            "adaptive_lr/loss_ema": self.loss_ema,
            "adaptive_lr/best_loss_ema": self.best_loss_ema,
            "adaptive_lr/no_improve_windows": self.no_improve_windows,
            "adaptive_lr/rising_windows": self.rising_windows,
            "adaptive_lr/cooldown_windows": self.cooldown_windows,
        }
        if _finite(loss_value):
            out["adaptive_lr/metric"] = float(loss_value)
        if _finite(update_ratio):
            out["adaptive_lr/update_ratio"] = float(update_ratio)
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", str(idx))
            out[f"adaptive_lr/scale/{name}"] = self.lr_scales[idx]
        if self.logger and decision.adjusted:
            try:
                self.logger.info(
                    "Adaptive LR %s: %s; scales=%s",
                    decision.state,
                    decision.reason,
                    [round(scale, 6) for scale in self.lr_scales],
                )
            except Exception:
                pass
        return out
