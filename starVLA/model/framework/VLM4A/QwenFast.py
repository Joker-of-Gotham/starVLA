# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-Fast Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen2.5 vision-language backbone
  - Unified action learning via next-token prediction (fast tokenizer)
  - Autoregressive action tokens derived from discretized / symbolized continuous actions

Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenFast
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenFastDefaultConfig:
    """QwenFast framework default parameters.

    Autoregressive discrete action prediction via FAST tokenizer.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenFast"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL with action special tokens) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (must include FAST action tokens)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (FAST tokenizer — discrete next-token prediction) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "FAST",
            # Local FAST processor/tokenizer. Override with STARVLA_FAST_TOKENIZER or YAML if needed.
            "fast_tokenizer_name": "playground/Pretrained_models/fast",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # How many future steps to predict
            "future_action_window_size": 15,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFast")
class Qwenvl_Fast(baseframework):
    """
    Multimodal vision-language-action model (FAST variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - FAST tokenizer for discretized / symbolized continuous action encoding
      - Autoregressive next-token prediction over action tokens

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenFastDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly predict future actions via next-token prediction (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]

        # step 0: map_raw_action_to_vlm_action
        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)  # List[str]

        # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )

        labels = qwen_inputs.pop("labels", None)
        input_ids = qwen_inputs.get("input_ids", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface.model.model(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
            )
            last_hidden = qwenvl_outputs[0]

        # Avoid HuggingFace's full-vocabulary CausalLM loss here. For Qwen3-VL-Action
        # it upcasts [B, L, vocab] logits to fp32 and can add >7GB per rank at
        # H200-sized batches. FAST supervision only needs the action-token slice.
        vlm_action_loss = self._fast_action_token_loss(
            hidden_states=last_hidden,
            input_ids=input_ids,
            labels=labels,
        )

        return {"action_loss": vlm_action_loss}

    def _fast_action_token_loss(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if labels is None or input_ids is None:
            return hidden_states.sum() * 0.0

        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        target_mask = (labels >= act_min) & (labels <= act_max)
        if not bool(target_mask.any()):
            return hidden_states.sum() * 0.0

        target_positions = target_mask.nonzero(as_tuple=False)
        batch_idx = target_positions[:, 0]
        token_pos = target_positions[:, 1]
        pred_pos = token_pos - 1
        valid = pred_pos >= 0
        if not bool(valid.any()):
            return hidden_states.sum() * 0.0

        batch_idx = batch_idx[valid]
        pred_pos = pred_pos[valid]
        targets = labels[target_mask][valid].long() - act_min

        pred_hidden = hidden_states[batch_idx, pred_pos, :]
        lm_head = self.qwen_vl_interface.model.lm_head
        action_weight = lm_head.weight[act_min : act_max + 1]
        action_bias = lm_head.bias[act_min : act_max + 1] if getattr(lm_head, "bias", None) is not None else None
        action_logits = F.linear(pred_hidden, action_weight, action_bias)
        return F.cross_entropy(action_logits.float(), targets)

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to obtain future actions (no diffusion sampling).
        # can be batch forward
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_length=2048,
            )
        # --- Extract and decode VLM action tokens to continuous actions. ---
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        normalized_actions = self._decode_fast_actions_or_fallback(batch_fast_action_token_idx)

        return {"normalized_actions": normalized_actions}

    def _action_output_shape(self) -> Tuple[int, int]:
        horizon = int(self.action_horizon)
        action_dim = int(self.config.framework.action_model.action_dim)
        return horizon, action_dim

    def _empty_action_batch(self, batch_size: int) -> np.ndarray:
        horizon, action_dim = self._action_output_shape()
        return np.zeros((int(batch_size), horizon, action_dim), dtype=np.float32)

    def _fit_decoded_action_shape(self, action: Any) -> np.ndarray:
        horizon, action_dim = self._action_output_shape()
        out = np.zeros((horizon, action_dim), dtype=np.float32)
        arr = np.asarray(action, dtype=np.float32)
        if arr.size == 0:
            return out
        if arr.ndim == 1:
            flat = arr.reshape(-1)
            take = min(flat.size, horizon * action_dim)
            out.reshape(-1)[:take] = flat[:take]
            return out

        arr = arr.reshape(arr.shape[0], -1)
        take_h = min(arr.shape[0], horizon)
        take_d = min(arr.shape[1], action_dim)
        out[:take_h, :take_d] = arr[:take_h, :take_d]
        return out

    def _warn_action_decode_fallback(self, missing_count: int, batch_size: int, reason: str | None = None) -> None:
        warn_count = int(getattr(self, "_fast_decode_fallback_warn_count", 0))
        if warn_count < 3:
            suffix = f" reason={reason}" if reason else ""
            logger.warning(
                "QwenFast generated no decodable FAST action tokens for "
                f"{missing_count}/{batch_size} eval samples; using zero-action fallback.{suffix}"
            )
        self._fast_decode_fallback_warn_count = warn_count + 1

    def _decode_fast_actions_or_fallback(self, batch_fast_token_ids: List[List[int]]) -> np.ndarray:
        batch_size = len(batch_fast_token_ids)
        normalized_actions = self._empty_action_batch(batch_size)
        valid = [(idx, seq) for idx, seq in enumerate(batch_fast_token_ids) if seq]
        if not valid:
            self._warn_action_decode_fallback(batch_size, batch_size)
            return normalized_actions

        valid_indices = [idx for idx, _ in valid]
        valid_sequences = [seq for _, seq in valid]
        try:
            decoded = self.action_model.fast_tokenizer.decode(valid_sequences)
        except Exception as exc:
            self._warn_action_decode_fallback(batch_size, batch_size, str(exc))
            return normalized_actions

        if isinstance(decoded, np.ndarray):
            if decoded.ndim == 2 and len(valid_indices) == 1:
                decoded_samples = [decoded]
            elif decoded.ndim >= 3:
                decoded_samples = [decoded[i] for i in range(min(decoded.shape[0], len(valid_indices)))]
            else:
                decoded_samples = [decoded]
        else:
            decoded_samples = list(decoded)

        for output_idx, decoded_action in zip(valid_indices, decoded_samples):
            normalized_actions[output_idx] = self._fit_decoded_action_shape(decoded_action)

        missing_count = batch_size - len(valid_indices)
        if missing_count > 0:
            self._warn_action_decode_fallback(missing_count, batch_size)
        return normalized_actions

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep all tokens falling within [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] in order of appearance.
        You may change it to "take only the first occurrence followed by continuous segment" as needed.
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # all action tokens
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append([])
                continue
            fast_ids = [t - act_min for t in seq if act_min <= t <= act_max]

            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on
        """
        return "".join(
            [f"<robot_action_{token}>" for token in tokens]
        )  # you should add <robot_action_{token}> to VLM as special tokens,


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_Fast(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # Untrained models haven't learned the action tokens, so predictions may be empty.
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
