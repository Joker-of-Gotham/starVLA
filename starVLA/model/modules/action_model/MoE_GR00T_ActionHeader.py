"""MoE variants for the GR00T/flow-matching action head.

These classes are intentionally small wrappers around the existing
``FlowmatchingActionHead`` so checkpoints trained with member-local
``QwenGR00T_MoE`` code can be loaded and evaluated from the shared repo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


class MoEActionDecoder(nn.Module):
    """Soft mixture-of-experts decoder for continuous action prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = int(num_experts)
        self.action_dim = int(action_dim)
        self.router = nn.Linear(input_dim, self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route_weights = F.softmax(self.router(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return torch.sum(route_weights.unsqueeze(-2) * expert_outputs, dim=-1)


class MoEFlowmatchingActionHead(FlowmatchingActionHead):
    """FlowmatchingActionHead with a MoE action decoder."""

    def __init__(self, full_config):
        super().__init__(full_config)
        action_cfg = full_config.framework.action_model
        moe_cfg = dict(action_cfg.moe_config) if hasattr(action_cfg, "moe_config") else {}
        num_experts = int(moe_cfg.get("num_experts", 4))
        expert_hidden_dim = int(moe_cfg.get("expert_hidden_dim", max(self.hidden_size // 4, 64)))
        self.action_decoder = MoEActionDecoder(
            input_dim=self.model.config.output_dim,
            hidden_dim=expert_hidden_dim,
            action_dim=self.action_dim,
            num_experts=num_experts,
        )


class AdaptivePlanner:
    """Convergence-gated early exit for flow/diffusion action inference."""

    def __init__(
        self,
        max_steps: int = 8,
        min_steps: int = 4,
        delta_threshold: float = 0.01,
        entropy_threshold: float = 0.5,
    ):
        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.delta_threshold = float(delta_threshold)
        self.entropy_threshold = float(entropy_threshold)

    @torch.no_grad()
    def should_stop(
        self,
        step: int,
        actions_old: torch.Tensor,
        actions_new: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> bool:
        if step < self.min_steps:
            return False
        delta = (actions_new - actions_old).abs().mean() / (actions_old.abs().mean() + 1e-8)
        if delta > self.delta_threshold:
            return False
        if entropy is not None and entropy.mean() > self.entropy_threshold:
            return False
        return True


class MoEAdaptiveFlowmatchingActionHead(MoEFlowmatchingActionHead):
    """MoE action head with adaptive denoising steps at inference."""

    def __init__(self, full_config):
        super().__init__(full_config)
        action_cfg = full_config.framework.action_model
        adaptive_cfg = dict(action_cfg.adaptive_planning) if hasattr(action_cfg, "adaptive_planning") else {}
        self.num_inference_timesteps = int(adaptive_cfg.get("max_steps", 8))
        self.planner = AdaptivePlanner(
            max_steps=adaptive_cfg.get("max_steps", 8),
            min_steps=adaptive_cfg.get("min_steps", 4),
            delta_threshold=adaptive_cfg.get("delta_threshold", 0.01),
            entropy_threshold=adaptive_cfg.get("entropy_threshold", 0.5),
        )

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        max_steps = self.planner.max_steps
        dt = 1.0 / max_steps
        state_features = self.state_encoder(state) if state is not None else None

        for step in range(max_steps):
            t_cont = step / float(max_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred_velocity = self.action_decoder(model_output)[:, -self.action_horizon :]
            actions_new = actions + dt * pred_velocity

            route_weights = F.softmax(self.action_decoder.router(model_output), dim=-1)
            entropy = -(route_weights * route_weights.clamp_min(1e-12).log()).sum(dim=-1)
            if self.planner.should_stop(step, actions, actions_new, entropy):
                return actions_new
            actions = actions_new
        return actions


def get_moe_action_model(config=None):
    return MoEFlowmatchingActionHead(full_config=config)


def get_moe_adaptive_action_model(config=None):
    return MoEAdaptiveFlowmatchingActionHead(full_config=config)
