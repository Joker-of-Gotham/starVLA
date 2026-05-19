# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.checkpointing import atomic_checkpoint_write, cfg_bool, cfg_get, cfg_str, checkpoint_summary, latest_state_checkpoint, parse_step_from_path, update_latest_state_link, update_topk_checkpoints
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.policy_runtime import PolicyRuntime
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = os.getenv("STARVLA_FAST_TOKENIZER", "playground/Pretrained_models/fast")
    return AutoProcessor.from_pretrained(fast_tokenizer, trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """Prepare co-training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    vlm_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vlm_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, vlm_train_dataloader


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, vlm_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vlm_train_dataloader = vlm_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.policy_runtime = PolicyRuntime.from_config(cfg)
        self.accelerator = accelerator

        self.completed_steps = 0
        self.vla_batches_seen = 0
        self.vlm_batches_seen = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so a later setup-step crash still
        # leaves a from_pretrained-able run dir behind.
        self._save_initial_configs()

        self._init_checkpointing()
        if self.resume_mode != "full":
            self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.vlm_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
                self.vlm_train_dataloader,
            )
        )

        self._resume_full_state_if_needed()
        if self.accelerator.is_main_process and self.policy_runtime.enabled:
            logger.info(f"Policy runtime enabled: {self.policy_runtime.describe()}")
        self._init_wandb()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"\U0001f4dd Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"\U0001f4ca Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directories and resolve resume mode."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        self.state_checkpoint_dir = os.path.join(self.config.output_dir, "accelerate_state")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.state_checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = cfg_str(self.config.trainer, "pretrained_checkpoint", None)
        explicit_resume = cfg_str(self.config.trainer, "resume_from_checkpoint", None)
        is_resume = cfg_bool(self.config.trainer, "is_resume", False)
        requested_mode = (cfg_str(self.config.trainer, "resume_mode", "auto") or "auto").lower()
        self.save_full_state = cfg_bool(self.config.trainer, "save_full_state", True)
        self.resume_from_checkpoint = pretrained_checkpoint
        self.pending_full_state_resume = None
        self.resume_mode = "none"

        if is_resume and requested_mode in {"auto", "full"}:
            state_checkpoint = None
            state_steps = 0
            if explicit_resume and os.path.isdir(explicit_resume):
                state_checkpoint = explicit_resume
                state_steps = parse_step_from_path(explicit_resume)
            else:
                state_checkpoint, state_steps = latest_state_checkpoint(self.state_checkpoint_dir)
            if state_checkpoint:
                self.pending_full_state_resume = state_checkpoint
                self.resume_from_checkpoint = state_checkpoint
                self.completed_steps = state_steps
                self.resume_mode = "full"
                logger.info(f"Prepared full-state resume from {state_checkpoint}, steps: {self.completed_steps}")
                return
            if requested_mode == "full":
                raise FileNotFoundError(
                    f"Full-state resume requested, but no Accelerate state checkpoint was found in {self.state_checkpoint_dir}"
                )

        if is_resume:
            if explicit_resume and os.path.isfile(explicit_resume):
                resume_from_checkpoint = explicit_resume
                self.completed_steps = parse_step_from_path(explicit_resume)
            else:
                resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self.resume_mode = "weights"
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return
            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = cfg_get(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            self.resume_mode = "pretrained"
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler for weight-only resumes."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _resume_full_state_if_needed(self):
        """Load model/optimizer/scheduler/RNG state after Accelerate.prepare()."""
        if not self.pending_full_state_resume:
            return
        self.accelerator.wait_for_everyone()
        self.accelerator.load_state(self.pending_full_state_resume)
        self.accelerator.wait_for_everyone()
        self.accelerator.print(f"Resumed full training state from: {self.pending_full_state_resume}")

    def _save_checkpoint(self, metrics: dict = None):
        """Save current training state."""
        save_format = getattr(self.config.trainer, "save_format", "pt")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        weight_checkpoint = None
        metrics = metrics or {}
        if self.accelerator.is_main_process:
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                weight_checkpoint = checkpoint_path + "_model.safetensors"
                atomic_checkpoint_write(weight_checkpoint, lambda tmp: save_file(state_dict, tmp))
            elif save_format == "pt":
                weight_checkpoint = checkpoint_path + "_pytorch_model.pt"
                atomic_checkpoint_write(weight_checkpoint, lambda tmp: torch.save(state_dict, tmp))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            self.accelerator.print(f"✅ Model checkpoint saved at {checkpoint_path}")

        self.accelerator.wait_for_everyone()

        state_checkpoint = None
        latest_state_ref = None
        if self.save_full_state:
            state_checkpoint = os.path.join(self.state_checkpoint_dir, f"steps_{self.completed_steps}")
            self.accelerator.save_state(output_dir=state_checkpoint)
            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"✅ Full training state saved at {state_checkpoint}")

        if self.accelerator.is_main_process:
            if state_checkpoint:
                latest_state_ref = update_latest_state_link(self.state_checkpoint_dir, state_checkpoint)
            topk = update_topk_checkpoints(
                output_dir=self.config.output_dir,
                completed_steps=self.completed_steps,
                weight_checkpoint=weight_checkpoint,
                state_checkpoint=state_checkpoint,
                metrics=metrics,
                k=int(getattr(self.config.trainer, "keep_top_k", 3)),
                keep_latest_state=latest_state_ref or state_checkpoint,
            )
            summary_data = checkpoint_summary(
                completed_steps=self.completed_steps,
                weight_checkpoint=weight_checkpoint,
                state_checkpoint=state_checkpoint,
                save_format=save_format,
                resume_mode=self.resume_mode,
            )
            summary_data.update(
                {
                    "metrics": metrics,
                    "loss": topk.get("current_loss"),
                    "topk_rank": topk.get("current_topk_rank"),
                    "kept_in_topk": topk.get("current_topk_rank") is not None,
                    "topk_index": os.path.join(self.config.output_dir, "topk_checkpoints.json"),
                    "latest_state": latest_state_ref,
                }
            )
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics.update(
                TrainerUtils.progress_metrics(
                    completed_steps=self.completed_steps,
                    batches_seen=self.vla_batches_seen,
                    dataloader=self.vla_train_dataloader,
                    max_train_steps=self.config.trainer.max_train_steps,
                    gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
                )
            )
            metrics.update(
                TrainerUtils.progress_metrics(
                    completed_steps=self.completed_steps,
                    batches_seen=self.vlm_batches_seen,
                    dataloader=self.vlm_train_dataloader,
                    max_train_steps=self.config.trainer.max_train_steps,
                    gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
                    prefix="vlm",
                )
            )
            metrics["global_batch_size"] = self.total_batch_size
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        resume_batches = self.completed_steps * max(int(self.accelerator.gradient_accumulation_steps or 1), 1)
        self.vla_batches_seen = resume_batches
        self.vlm_batches_seen = resume_batches
        self.vla_iter = self._iterator_with_resume_skip(self.vla_train_dataloader, "vla", resume_batches)
        self.vlm_iter = self._iterator_with_resume_skip(self.vlm_train_dataloader, "vlm", resume_batches)

    def _iterator_with_resume_skip(self, dataloader, name: str, consumed_batches: int | None = None):
        if self.completed_steps <= 0 or self.resume_mode == "none":
            return iter(dataloader)
        try:
            dataloader_len = len(dataloader)
        except TypeError:
            return iter(dataloader)
        if not dataloader_len:
            return iter(dataloader)
        skip_batches = int(consumed_batches if consumed_batches is not None else self.completed_steps) % dataloader_len
        if skip_batches <= 0:
            return iter(dataloader)
        logger.info(f"Skipping {skip_batches} {name} batches to align resume position.")
        return iter(self.accelerator.skip_first_batches(dataloader, skip_batches))

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        self.vla_batches_seen += 1

        try:
            batch_vlm = next(self.vlm_iter)
        except StopIteration:
            if not hasattr(self, "vlm_epoch_count"):
                self.vlm_epoch_count = 0
            self.vlm_iter, self.vlm_epoch_count = self._reset_dataloader(self.vlm_train_dataloader, self.vlm_epoch_count)
            batch_vlm = next(self.vlm_iter)
        self.vlm_batches_seen += 1

        return batch_vla, batch_vlm

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_info = TrainerUtils.progress_metrics(
                    completed_steps=self.completed_steps,
                    batches_seen=self.vla_batches_seen,
                    dataloader=self.vla_train_dataloader,
                    max_train_steps=self.config.trainer.max_train_steps,
                    gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
                )
                postfix = {
                    "epoch": f"{progress_info.get('epoch', '-')}/{progress_info.get('total_epochs', '-')}",
                    "data_times": f"{t_end_data - t_start_data:.3f}",
                    "model_times": f"{t_end_model - t_start_model:.3f}",
                }
                progress_bar.set_postfix(postfix)

            if not self.accelerator.sync_gradients:
                continue

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint(step_metrics)
                dist.barrier()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate action prediction with current model."""
        if self.accelerator.is_main_process:
            examples, _ = self._get_next_batch()
            actions = [example["action"] for example in examples]

            output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)
            normalized_actions = output_dict["normalized_actions"]

            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            steps_per_epoch = TrainerUtils.dataloader_len(self.vla_train_dataloader)
            vlm_steps_per_epoch = TrainerUtils.dataloader_len(self.vlm_train_dataloader)
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  VLA steps per epoch = {steps_per_epoch or 'unknown'}")
            logger.info(f"  VLM steps per epoch = {vlm_steps_per_epoch or 'unknown'}")
            if steps_per_epoch:
                total_epochs = (
                    self.config.trainer.max_train_steps
                    * max(int(self.accelerator.gradient_accumulation_steps or 1), 1)
                    / steps_per_epoch
                )
                logger.info(f"  Estimated VLA total epochs = {total_epochs:.4f}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm):
        """Execute single training step."""
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss, policy_metrics = self.policy_runtime.apply_action_loss(
                    action_loss=action_loss,
                    output_dict=output_dict,
                    model=self.model,
                    batch=batch_vla,
                )
            self.accelerator.backward(total_loss)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unwrapped = self.accelerator.unwrap_model(self.model)
                vlm_output = unwrapped.qwen_vl_interface(**batch_vlm)
                vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
                vlm_loss, vlm_policy_metrics = self.policy_runtime.apply_vlm_loss(
                    vlm_loss=vlm_loss,
                    model=self.model,
                    batch=batch_vlm,
                )
            self.accelerator.backward(vlm_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced.
            # See train_starvla.py for full explanation.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vlm_loss": vlm_loss.item(),
                }
            )
            log_dict.update(policy_metrics)
            log_dict.update({f"vlm_{key}": value for key, value in vlm_policy_metrics.items()})

        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                atomic_checkpoint_write(os.path.join(final_checkpoint, "model.safetensors"), lambda tmp: save_file(state_dict, tmp))
            elif save_format == "pt":
                atomic_checkpoint_write(os.path.join(final_checkpoint, "pytorch_model.pt"), lambda tmp: torch.save(state_dict, tmp))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        self.accelerator.wait_for_everyone()
        if getattr(self, "save_full_state", True):
            final_state = os.path.join(self.config.output_dir, "final_state")
            self.accelerator.save_state(output_dir=final_state)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                logger.info(f"Final full training state saved at {final_state}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    TrainerUtils.configure_torch_runtime(logger)
    cfg = TrainerUtils.apply_interaction_safety_caps(cfg, logger)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vlm_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vlm_train_dataloader=vlm_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
