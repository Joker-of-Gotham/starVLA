# StarVLA Interaction Layer

This directory provides a tmux-managed control layer for StarVLA. It does not
replace the existing training or evaluation code; it builds commands around the
current `starVLA/training/*.py` and `examples/*/eval_files/*` entry points.

Main entry:

```bash
bash interaction/bin/starvla-interact.sh
```

Useful non-interactive commands:

```bash
# Idempotent GPU repair. Healthy envs skip package reinstall and editable
# wheel rebuild; use --force-reinstall only for deliberate package repair.
bash interaction/bin/starvla-gpu-fix.sh --num-gpus 8

# Full environment/path/import/action-head check.
bash interaction/bin/starvla-interact.sh check
bash interaction/bin/starvla-interact.sh paths

# Inspect 8-GPU H200 topology, NCCL profile, power/clock visibility, and
# bandwidth-test readiness before a real training launch.
bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8

# Real tmux CPU-only orchestration self-test, including a second tmux window.
bash interaction/bin/starvla-interact.sh selftest --show-log

# Show the exact command without launching it.
bash interaction/bin/starvla-interact.sh train --preset libero_vla --gpus auto --num-gpus 8 --dry-run

# Launch a new 8xH200 training run in tmux with the automatic performance profile.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode new --gpus auto --num-gpus 8 --perf-profile auto --throughput-profile auto

# Continue an existing run_id from its latest full Accelerate/DeepSpeed state.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode continue --run-id <run_id>

# Continue the newest run inside the same <model>-<action_head>-<dataset> group.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode continue

# Resume from an explicit full state directory or weight file.
bash interaction/bin/starvla-interact.sh train --preset bridge_vla --resume-from-checkpoint <path> --resume-mode auto

# Skip the launch confirmation when running from scripts.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode new --yes

# List and monitor jobs.
bash interaction/bin/starvla-interact.sh list
bash interaction/bin/starvla-interact.sh monitor <job-name-or-run-dir> --once
bash interaction/bin/starvla-interact.sh monitor <job-name-or-run-dir> --tail-lines 18

# Attach or stop the underlying tmux session.
bash interaction/bin/starvla-interact.sh attach <job-name-or-run-dir>
bash interaction/bin/starvla-interact.sh stop <job-name-or-run-dir>
```

Training outputs are saved under a first-level experiment group:

`/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/<model>-<action_head>-<dataset>/<run_id>/`

- `checkpoints/steps_<N>_pytorch_model.pt` or `checkpoints/steps_<N>_model.safetensors`: top-3 model weights by lowest loss
- `topk_checkpoints.json`: sorted top-k index with loss, step, weight path, state path, and metrics
- `accelerate_state/steps_<N>/`: full resume state from `accelerator.save_state()` for top-k states plus the latest state
- `accelerate_state/latest`: symlink or marker for continuing the current run
- `final_model/`: final model weights
- `final_state/`: final full training state
- `summary.jsonl`: per-checkpoint index containing weight/state paths
- `config.full.yaml` and `config.yaml`: complete and accessed config snapshots

Use `--keep-top-k <K>` to change the retained weight count. The default is
`3`. `--run-mode new` starts a new run_id under the same experiment group;
`--run-mode continue` keeps the same run_id and resumes from
`accelerate_state/latest` or the latest `accelerate_state/steps_<N>/`.

Before a tmux train/eval launch, the CLI shows an interactive confirmation
table with model, action head, dataset, experiment group, run_id, GPUs, top-k,
and output paths. Type `yes` or `y` to start. Use `--yes` for non-interactive
automation.

In the interactive `launch_train` flow, choose `Run mode -> continue` to resume.
The menu looks for the latest run under the selected
`<model>-<action_head>-<dataset>` experiment group. Leave `run_id` blank to
resume that latest run, or type a specific existing `run_id`. Use `Resume mode
-> auto` for normal continuation; it prefers the full Accelerate/DeepSpeed
state and falls back to weights when needed.

The training launcher now runs a preflight table before tmux startup. It checks
the trainer script, config YAML, dataset root, base model, Accelerate config,
and output-root writability so missing paths fail before a long training job is
created. If no GPU is selected, training refuses to launch unless `--allow-cpu`
is passed for an intentional tiny smoke run.

Interaction job files are stored under
`/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/<timestamp>_<job>/`
when writable. If that project data directory is read-only in the current shell,
self-tests and launch metadata fall back to `interaction/runs/<timestamp>_<job>/`
and record the fallback in `job.json`:

- `job.json`: structured metadata
- `*.command.sh`: generated command
- `*.runner.sh`: tmux runner with no-proxy activation
- `*.log`: live log parsed by `monitor`
- `environment.json`: startup environment/GPU/topology snapshot
- `events.jsonl`: started/phase/heartbeat/failed/finished events for disconnected monitoring

The launcher automatically detects available GPUs through `nvidia-smi` for
`--gpus auto`, respects `CUDA_VISIBLE_DEVICES`/`SLURM_JOB_GPUS`, prioritizes
H200 cards with the most free memory, and by default filters for at least 60000
MB free memory per GPU before falling back to all visible cards. It then passes
the selected devices through `CUDA_VISIBLE_DEVICES` and sets
`accelerate launch --num_processes` to the number of selected GPUs. The default
mixed precision is BF16, with NCCL and PyTorch nonblocking error-handling
environment variables enabled in the tmux runner.

For 8 selected H200 GPUs, `--perf-profile auto` resolves to `h200_8gpu` when
the local `nvidia-smi` inventory confirms the hardware. That profile uses
`interaction/config/accelerate_h200_zero2.yaml`, which points to
`interaction/config/ds_h200_zero2.json`: ZeRO-2, BF16, no CPU offload,
`overlap_comm`, `reduce_scatter`, `contiguous_gradients`, larger 1e9
communication buckets, and reduced DeepSpeed print frequency. It also sets
safe NCCL/PyTorch defaults such as high-priority NCCL streams, async error
handling, nonblocking waits, topology dump, and heartbeat monitoring. NCCL's
own topology selection is preserved unless NVLink is detected, in which case
the H200 profile defaults `NCCL_P2P_LEVEL=NVL` for intra-node P2P.

Useful performance overrides:

```bash
# Force aggressive H200 batch/dataloader settings. For Qwen3.5-2B + PI this
# raises Calvin VLA per-device batch above the old low preset and enables
# persistent prefetched workers.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --framework QwenPI --model qwen35_2b --gpus 0,1,2,3,4,5,6,7 --perf-profile h200_8gpu --throughput-profile h200_saturated

# If the aggressive profile hits OOM, keep the H200 dataloader profile but cap
# the batch explicitly.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --framework QwenPI --model qwen35_2b --gpus 0,1,2,3,4,5,6,7 --perf-profile h200_8gpu --throughput-profile h200_saturated --vla-batch-size 16

# Force the H200 profile when running on a known 8xH200 node.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --gpus 0,1,2,3,4,5,6,7 --perf-profile h200_8gpu

# Increase NCCL startup detail for one run.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --num-gpus 8 --nccl-debug INFO

# Manually pin fabric/network choices only when the cluster requires it.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --num-gpus 8 --nccl-socket-ifname '=ib0' --nccl-ib-hca '=mlx5_0:1,=mlx5_1:1'

# Add any extra override without changing code.
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --num-gpus 8 --nccl-env NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=64
```

The throughput profile is separate from the NCCL/DeepSpeed performance profile.
It controls per-device batch defaults plus DataLoader `num_workers`,
`prefetch_factor`, `pin_memory`, `persistent_workers`, and CPU thread caps.
Training logs and the monitor now report `epoch`, `epoch_index`,
`epoch_batch`, `steps_per_epoch`, `total_epochs`, and `data_batches_seen`, and
the tqdm postfix shows `epoch=current/total` while the job is running.

The monitor uses a Rich live dashboard when available and an ANSI in-place
fallback otherwise, so repeated refreshes update the same terminal area instead
of continuously scrolling. It parses training metrics, progress, top-k
checkpoint state, GPU heartbeat snapshots with memory/util/power/clocks, recent log tails, failed runner
commands, and common root-cause patterns such as CUDA OOM, NCCL/distributed
failures, missing paths, permission errors, disk quota, import errors, NaN/Inf
loss, DataLoader worker failures, and eval server/client port issues.
