# StarVLA Offline Environment Package

This package is staged inside the repo because the requested final path is not writable from the current CPU sandbox.

Primary offline install path:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA/envSet/starvla_env
bash scripts/install_offline.sh
```

To force replacing the current env from the packaged snapshot:

```bash
bash scripts/install_offline.sh --force-restore
```

GPU one-click repair, with no proxy and no network access:

```bash
bash scripts/gpu_oneclick_fix.sh --num-gpus 8
```

The same repair can be launched from the repo after the overlay is installed:

```bash
bash interaction/bin/starvla-gpu-fix.sh --num-gpus 8
```

To activate after install:

```bash
source scripts/activate_offline_env.sh
```

The installer now also refreshes the packaged `interaction/` overlay, including
the H200 ZeRO-2 Accelerate/DeepSpeed configs, `interaction/core/perf.py`,
`perf-check`, and the tmux binary validation fallback. Use
`--skip-interaction-overlay` only if you deliberately want to keep a different
local `interaction/` tree.

H200 performance diagnostics after activation:

```bash
bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --gpus auto --num-gpus 8 --perf-profile auto
```

The offline package installs NCCL `*_perf` wrappers into
`env/starvla-py310/bin`, so `all_reduce_perf` is available to `perf-check` even
when the GPU node cannot access the network.
`p2pBandwidthLatencyTest` is also bundled from NVIDIA CUDA Samples v12.4.
`nvbandwidth` is an optional deeper diagnostic; if the GPU node does not already
provide it, `perf-check` reports that as an OK optional-missing state rather
than a launch blocker.
The official `nvbandwidth` v0.9 source is included under
`performance_tools/src/nvbandwidth`; if the GPU node has Boost program_options
development files, it can be built offline with:

```bash
bash scripts/build_optional_nvbandwidth.sh
bash scripts/install_perf_tools.sh
```

To sync this package to the requested final location when that path is writable:

```bash
bash scripts/sync_to_requested_target.sh
```

Layout:

- `archives/`: full `env/starvla-py310` snapshot, including the fixed `flash_attn`.
- `wheelhouse/flash_attn/`: verified FlashAttention wheel for Torch 2.6.0 + CUDA 12.4 + Python 3.10.
- `wheelhouse/core/`: partial wheels downloaded or built during packaging; the full env archive is the authoritative offline source.
- `requirements/`: frozen CPU environment requirements and download lists.
- `system_tools/tmux/`: tmux binary wrapper and required shared libraries for GPU nodes without tmux in PATH.
- `performance_tools/`: packaged NCCL test binaries, wrappers, and nccl-tests source for GPU-side rebuilds.
- `repo_overlay/interaction/`: current interaction H200 performance-control layer, excluding run logs and Python caches.
- `etc/h200_perf_env.sh`: optional shell profile for manual H200/NCCL diagnostics.
- `reports/`: CPU self-check and public environment scan reports.
- `scripts/`: one-click install, check, activation, and sync helpers.
