# StarVLA SII SummerCamp 使用与维护说明

本文档面向本分支的训练、后训练、评估、ensemble 和仓库维护。顶层 README 保留上游 StarVLA 项目介绍；本文件解释当前分支新增的工程化能力、目录规范和推荐操作方式。赛题总结和答辩材料底稿见 [SUMMERCAMP_REPORT.md](SUMMERCAMP_REPORT.md)。

## 1. 仓库结构

```text
starVLA/
  starVLA/
    dataloader/                         数据集、LeRobot/GR00T 数据格式、registry
    model/
      framework/VLM4A/                  Qwen/Gemma/Cosmos/Wan 等 VLA framework
      framework/WM4A/                   世界模型到动作预测的 framework
      modules/action_model/             OFT/PI/FAST/GR00T/MoE 等 action expert
      modules/vlm/                      VLM 接口
      modules/world_model/              Cosmos/Wan 世界模型模块
    training/                           train_starvla、cotrain、vlm-only 训练入口
  examples/
    calvin/                             CALVIN 训练配置、数据 registry、评估脚本
    LIBERO/, Robotwin/, Robocasa_*/     其它 benchmark 的配置和评估入口
  deployment/model_server/              WebSocket policy server/client、服务端 norm processor
  interaction/                          交互式训练/评估/监控/ensemble/tmux 管理层
  docs/                                 中文说明和上游架构文档
```

`examples/` 当前不能删除。它不是纯展示样例，而是保存了每个 benchmark 的数据 registry、训练 YAML、评估脚本和环境适配器。删除后 CALVIN、LIBERO、RoboTwin、RoboCasa 等入口会失效。

`interaction/bin/` 已清理为长期可复用脚本：

```text
starvla-interact.sh             主入口，训练/评估/监控/ensemble 都从这里走
starvla-disk-guard-resume.sh    公共盘空间不足时自动停训，空间恢复后自动续训
starvla-clean-gpu-orphans.sh    清理残留 accelerate/python GPU 进程
starvla-force-stop.sh           强制停止交互式 tmux job
starvla-gpu-fix.sh              快速检查 GPU/NCCL 环境
```

一次性的公共目录迁移、固定 LZH 复制、固定 CALVIN ensemble wrapper 已删除。对应动作请使用通用 `starvla-interact.sh ensemble`、普通 `cp/rsync` 或交互式训练/评估入口完成。

## 2. 不进入 Git 的内容

以下内容应保留在本地或公共数据盘，不提交到 Git：

- Python/conda/离线环境：`env/`、`.venv/`、`envSet/`
- 数据集、预训练模型、checkpoint：`playground/`、`data/`、`results/`
- 交互式运行产物：`interaction/runs/`、`interaction/ensembles/`
- 日志、pid、缓存：`*.log`、`*.pid`、`__pycache__/`
- 大模型权重和二进制中间文件：`*.pt`、`*.npy`、`*.parquet`、`*.mp4`

公共 checkpoint 推荐放在：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/<member>/
```

## 3. 数据怎么放

仓库内默认用相对路径 `playground/Datasets/...`。在当前机器上推荐把真实数据放在仓库的 `playground/Datasets` 或公共盘，然后在 `interaction/config/policies.json` 或命令行覆盖 `data_root_dir`。

推荐结构：

```text
starVLA/playground/Datasets/
  calvin/
    training/
    validation/
    task_ABC_D/
  libero/
  bridge/
  oxe/
  robotwin/
  robocasa_gr1/
```

CALVIN 默认数据入口：

```text
examples/calvin/train_files/starvla_train_calvin.yaml
examples/calvin/train_files/data_registry/data_config.py
examples/calvin/eval_files/eval_calvin.py
examples/calvin/eval_files/eval_calvin.sh
```

当前交互式 catalog 中已注册的数据集：

```text
libero_all
calvin_abc
calvin_abc_d
oxe_bridge
oxe_bridge_rt1
robotwin_all
robotwin_all_50
robocasa_gr1
robocasa365_smoke
vlm_robot
```

新增数据集时至少需要做三件事：

1. 在 `interaction/config/policies.json` 的 `datasets` 增加 dataset key，写清 `data_root_dir`、`data_mix`、`action_dim`、`state_dim`、`horizon`、normalization group。
2. 在对应 `examples/<benchmark>/train_files/...` 注册 data mix，保证 dataloader 能通过 key 找到真实文件。
3. 如果是新机器人或新 action space，给 `dataset_statistics.json` 准备正确的 action/state normalization，避免评估阶段反归一化错误。

## 4. 模型怎么放

预训练模型推荐放在：

```text
starVLA/playground/Pretrained_models/
  Qwen3.5-0.8B/
  Qwen3.5-0.8B-Action/
  Qwen3.5-2B/
  Qwen3.5-2B-Action/
  Qwen3.5-4B/
  Qwen3.5-4B-Action/
  Qwen3.5-9B/
  Qwen3.5-9B-Action/
  Qwen3-VL-4B-Instruct/
  Qwen3-VL-4B-Instruct-Action/
  Cosmos-Predict2-2B-Video2World/
```

当前 catalog 中已注册的 base model：

```text
qwen35_0_8b
qwen35_0_8b_action
qwen35_2b
qwen35_2b_action
qwen35_4b
qwen35_4b_action
qwen35_9b
qwen35_9b_action
qwen3vl_4b
qwen3vl_4b_action
cosmos_predict2
```

新增 base model 时，在 `interaction/config/policies.json` 的 `base_models` 中增加 key，并写清：

- `path`：模型目录或 HuggingFace/local path
- `family`：例如 `qwen`、`cosmos`、`wan`、`gemma`
- `default_framework` 或 family route：用于把抽象 expert 自动解析到具体 framework
- 是否是 action-token 版本：FAST/action-token 路线需要 embedding/head 与 tokenizer 对齐

## 5. checkpoint 怎么放

训练输出统一保存在：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/<experiment_group>/<run_id>/
```

典型结构：

```text
config.yaml                         运行配置快照
config.full.yaml                    完整配置快照
dataset_statistics.json             normalization / unnormalization stats
summary.jsonl                       checkpoint 索引和训练指标
topk_checkpoints.json               top-k checkpoint 排名
checkpoints/steps_<N>_pytorch_model.pt
accelerate_state/steps_<N>/         DeepSpeed/Accelerate 完整恢复状态
final_model/pytorch_model.pt
final_state/
```

使用规则：

- 只做评估或 ensemble：使用 `checkpoints/steps_<N>_pytorch_model.pt` 或 `final_model/pytorch_model.pt`。
- 继续同一个 run 训练：需要 `accelerate_state/steps_<N>/`，命令使用 `--run-mode continue`。
- 基于已有模型另开新 run 后训练：使用 `--posttrain-from /path/to/pytorch_model.pt`，不要和 `--run-mode continue` 混用。

公共成员目录推荐结构：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/<member>/
  runs/<run_name>/
    checkpoints/
    final_model/
    config.yaml
    dataset_statistics.json
  reports/<eval_name>/
    results.json
```

如果需要把某个本地权重复制到公共目录，使用标准目录和清晰文件名，例如：

```bash
mkdir -p /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/LZH/runs/<run_name>/checkpoints
cp /path/to/steps_5500_pytorch_model.pt \
  /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/LZH/runs/<run_name>/checkpoints/
mkdir -p /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/LZH/runs/<run_name>/final_model
ln -sf ../checkpoints/steps_5500_pytorch_model.pt \
  /inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/LZH/runs/<run_name>/final_model/pytorch_model.pt
```

## 6. 交互式入口

主入口：

```bash
bash interaction/bin/starvla-interact.sh
```

常用命令：

```bash
# 环境、路径、导入、action head 检查
bash interaction/bin/starvla-interact.sh check
bash interaction/bin/starvla-interact.sh paths

# 查看完整组合空间
bash interaction/bin/starvla-interact.sh catalog --kind all

# H200/NCCL/带宽工具检查
bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8

# 启动训练
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode new --gpus auto --num-gpus 8

# 继续同一 experiment group 下最新 run
bash interaction/bin/starvla-interact.sh train --preset calvin_vla --run-mode continue

# 监控/进入/停止 tmux job
bash interaction/bin/starvla-interact.sh list
bash interaction/bin/starvla-interact.sh monitor <job-name-or-run-dir>
bash interaction/bin/starvla-interact.sh attach <job-name-or-run-dir>
bash interaction/bin/starvla-interact.sh stop <job-name-or-run-dir>
```

## 7. 自由组合流程

本分支把训练配置拆成五层：

```text
datasets -> training policy -> base model -> action expert -> structure policy
```

组合示例：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --training-policy T01 T06 T07 \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --structure-policy S01 S07 \
  --gpus auto --num-gpus 8
```

关键点：

- dataset 决定 `action_dim`、`state_dim`、`action_horizon`、normalization group。
- base model 与 action expert 解耦。抽象 expert 如 `PI/OFT/GR00T/FAST` 会按 base family 解析到具体 framework。
- Training Policy 和 Structure Policy 不只记录在 metadata，也会通过 `trainer.policy_runtime.*` 进入训练运行时。
- 当前支持 28 个 Training Policy：`T01` 到 `T28`。
- 当前支持 32 个 Structure Policy：`S01` 到 `S32`。
- 当前 action expert catalog 包含抽象 expert 和具体 framework：`PI`、`OFT`、`GR00T`、`FAST`、`Adapter`、`Dual`、`ABot_M0`、`LangForce`、`QwenPI`、`QwenFM`、`QwenPI_v3`、`QwenOFT`、`QwenGR00T`、`QwenFast`、`QwenAdapter`、`QwenDual`、`Gemma4PI`、`Gemma4GR00T`、`WanPI`、`WanOFT`、`WanGR00T`、`CosmoPredict2PI`、`CosmoPredict2OFT`、`CosmoPredict2GR00T`、`CosmosGR00T`、`InternVLA-M1`。

## 8. 后训练

从已有 checkpoint 后训练：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --training-policy T20 T21 T24 T27 T28 \
  --structure-policy S01 S26 S27 S30 \
  --posttrain-from /path/to/checkpoints/steps_60000_pytorch_model.pt \
  --gpus auto --num-gpus 8
```

推荐策略：

- `T20`：失败任务 replay / hard-task curriculum
- `T21`：anti-forgetting，混 replay / teacher KL / hidden distillation
- `T24`：offline RL / advantage-weighted BC
- `T27`：trajectory ranking / DPO-style preference
- `T28`：EMA/SWA/checkpoint averaging/final distillation

后训练推荐先做 dry-run：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --training-policy T20 T21 \
  --structure-policy S01 S26 S27 \
  --posttrain-from /path/to/pytorch_model.pt \
  --dry-run
```

确认输出目录、framework、action/state/horizon、init checkpoint 都正确后，再去掉 `--dry-run`。

## 9. H200 高速训练

默认 8 卡 H200 推荐：

```bash
bash interaction/bin/starvla-interact.sh train \
  --preset calvin_vla \
  --gpus 0,1,2,3,4,5,6,7 \
  --perf-profile h200_8gpu \
  --throughput-profile h200_saturated
```

H200 profile 会启用：

- BF16 mixed precision
- DeepSpeed ZeRO-2，无 CPU offload
- NCCL async error handling、nonblocking wait、heartbeat monitoring
- NVLink 节点默认 `NCCL_P2P_LEVEL=NVL`
- 更大的通信 bucket、prefetch/persistent dataloader workers
- 训练监控中显示 GPU memory、util、power、clock、step time 和 top-k checkpoint

如果机器上已有进程占显存，preflight 会显示具体 PID、显存和 GPU util。是否继续由用户判断；代码不会因为还有其它进程就强行阻止启动。显存不足时优先调小 `--vla-batch-size` 或换 `--throughput-profile balanced`，不要直接删除 checkpoint。

## 10. 磁盘守护恢复训练

公共盘空间波动时使用：

```bash
bash interaction/bin/starvla-disk-guard-resume.sh \
  --run-dir /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/<run_dir> \
  --checkpoint-dir /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/<group>/<run_id> \
  --watch-dir /inspire/qb-ilm2/project/26summer-camp-10/public \
  --stop-free-gb 30 \
  --resume-free-gb 50
```

逻辑：

- 空间低于 `stop-free-gb` 时终止训练进程，避免写坏权重。
- 空间恢复到 `resume-free-gb` 后，从完整 `accelerate_state` 自动续训。
- 过小的疑似损坏 checkpoint 会被忽略。
- 循环直到 `summary.jsonl` 显示训练达到 `max_train_steps`。

## 11. 评估

CALVIN websocket eval：

```bash
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu all \
  --trials 1000
```

本分支修复了一个常见问题：client 请求 `unnorm_key=franka`，但 checkpoint 只有 `new_embodiment`。服务端现在会在单 key checkpoint 下自动映射到唯一可用 key；客户端也会根据 server metadata 做同样修正。

如果评估失败，优先检查：

- `logs/server*.log`：模型加载、OOM、norm key、policy inference error
- `logs/client*.log`：环境初始化、websocket 响应、rollout 错误
- `calvin_eval/workers/*/sequence_progress.json`：每个 worker 的当前进度
- `calvin_eval/merged_results.json`：合并后的最终统计

## 12. Ensemble

构建 CALVIN ultimate model soup：

```bash
bash interaction/bin/starvla-interact.sh ensemble \
  --recipe calvin_ultimate_moe \
  --method weighted_soup \
  --output-dir /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/ensembles/calvin_ultimate_moe_soup \
  --overwrite \
  --yes
```

输出结构和普通 run 一致：

```text
config.yaml
dataset_statistics.json
ensemble.json
checkpoints/steps_ensemble_pytorch_model.pt
```

输出 checkpoint 可以直接评估：

```bash
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/ensembles/calvin_ultimate_moe_soup/checkpoints/steps_ensemble_pytorch_model.pt \
  --server-gpu all \
  --trials 1000
```

也可以作为后训练初始化：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --training-policy T20 T21 T28 \
  --structure-policy S01 S30 \
  --posttrain-from /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/ensembles/calvin_ultimate_moe_soup/checkpoints/steps_ensemble_pytorch_model.pt \
  --gpus auto --num-gpus 8
```

默认 `calvin_ultimate_moe` recipe 使用 GTY 60k、GTY MoE posttrain 95k、WMH adaptive MoE 15k checkpoint 和 WMH final model。默认权重偏向两个 GTY checkpoint，同时保留 WMH 作为多样性来源：`0.38,0.42,0.10,0.10`。可用 `--method uniform_soup` 或 `--weights` 覆盖。

## 13. 提交规范

提交前建议：

```bash
git status --short
python -m py_compile \
  interaction/starvla.py \
  interaction/core/ensemble.py \
  deployment/model_server/policy_wrapper.py \
  deployment/model_server/policy_norm_processor.py \
  deployment/model_server/tools/websocket_policy_client.py \
  examples/LIBERO/eval_files/model2libero_interface.py \
  examples/calvin/eval_files/eval_calvin.py

bash -n interaction/bin/starvla-interact.sh
bash -n interaction/bin/starvla-disk-guard-resume.sh
bash -n interaction/bin/starvla-clean-gpu-orphans.sh
bash -n interaction/bin/starvla-force-stop.sh
bash -n interaction/bin/starvla-gpu-fix.sh
```

当前分支目标名：

```text
star_vla_sii_summercamp
```
