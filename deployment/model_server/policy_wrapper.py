# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        self._state_adaptation_warned: set[tuple[int, int]] = set()
        runtime_cfg = model_cfg.get("trainer", {}).get("policy_runtime", {})
        structure_policies = str(runtime_cfg.get("structure_policies", "") or "")
        self._safety_projection = (
            "S27" in {item.strip() for item in structure_policies.replace(",", " ").split()}
            or os.getenv("STARVLA_POLICY_SAFETY_PROJECTION", "0").strip().lower() in {"1", "true", "yes", "on"}
        )

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]
        
        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        action_model_cfg = self._model_cfg.get("framework", {}).get("action_model", {})
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "action_dim": action_model_cfg.get("action_dim"),
            "state_dim": action_model_cfg.get("state_dim"),
            "action_horizon": action_model_cfg.get("action_horizon"),
            "future_action_window_size": action_model_cfg.get("future_action_window_size"),
            "action_model_type": action_model_cfg.get("action_model_type"),
            "safety_projection": self._safety_projection,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def _expected_state_dim(self) -> int | None:
        action_model_cfg = self._model_cfg.get("framework", {}).get("action_model", {})
        value = action_model_cfg.get("state_dim")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _adapt_example_state(self, example: dict) -> dict:
        if os.getenv("STARVLA_POLICY_STATE_ADAPT", "1").strip().lower() in {"0", "false", "no", "off"}:
            return example
        expected = self._expected_state_dim()
        if expected is None or "state" not in example:
            return example
        state = np.asarray(example["state"], dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if state.shape[-1] == expected:
            return example

        actual = int(state.shape[-1])
        if actual > expected:
            adapted = state[..., :expected]
        else:
            pad_width = [(0, 0)] * state.ndim
            pad_width[-1] = (0, expected - actual)
            adapted = np.pad(state, pad_width, mode="constant")

        key = (actual, expected)
        if key not in self._state_adaptation_warned:
            logging.warning(
                "PolicyServerWrapper adapted example state_dim from %d to %d for checkpoint %s. "
                "Set STARVLA_POLICY_STATE_ADAPT=0 to disable this guard.",
                actual,
                expected,
                self._ckpt_path,
            )
            self._state_adaptation_warned.add(key)
        new_example = dict(example)
        new_example["state"] = adapted
        return new_example

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        prepared_examples = [self._adapt_example_state(example) for example in examples]

        with torch.inference_mode():
            out = self._framework.predict_action(examples=prepared_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        if self._safety_projection:
            low = float(os.getenv("STARVLA_POLICY_ACTION_LOW", "-1000.0"))
            high = float(os.getenv("STARVLA_POLICY_ACTION_HIGH", "1000.0"))
            unnorm = np.nan_to_num(unnorm, nan=0.0, posinf=high, neginf=low)
            unnorm = np.clip(unnorm, low, high)
        return {"actions": unnorm}
