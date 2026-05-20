# StarVLA SII SummerCamp 使用与维护说明

本文档面向本分支的训练、后训练、评估、ensemble 和仓库维护。顶层 README 保留上游 StarVLA 项目介绍；本文件解释当前分支新增的工程化能力和推荐使用方式。

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

`examples/` 当前不能删除。它不是纯展示样例，而是保存了每个 benchmark 的数据 registry、训练 YAML、评估脚本和环境适配器。删除后 CALVIN/LIBERO/RoboTwin/RoboCasa 等入口会失效。

## 2. 不进入 Git 的内容

以下内容应保留在本地或公共数据盘，不提交到 Git：

- Python/conda/离线环境：`env/`、`.venv/`、`envSet/`
- 数据集、预训练模型、checkpoint：`playground/`、`data/`、`results/`
- 交互式运行产物：`interaction/runs/`、`interaction/ensembles/`
- 日志、pid、缓存：`*.log`、`*.pid`、`__pycache__/`

公共 checkpoint 推荐放在：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/<member>/
```

## 3. 交互式入口

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

## 4. 自由组合流程

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
- base model 与 action expert 解耦，抽象 expert 如 `PI/OFT/GR00T` 会按 base family 解析到具体 framework。
- Training Policy 和 Structure Policy 不只记录在 metadata，也会通过 `trainer.policy_runtime.*` 进入训练运行时。
- 已训练模型做后训练时使用 `--posttrain-from`，不要把它和 `--run-mode continue` 混用。`continue` 是恢复同一个 run 的优化器/调度器/RNG 状态；`posttrain-from` 是新 run 初始化权重。

## 5. 后训练

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

## 6. H200 高速训练

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

如果机器上已有进程占显存，preflight 会显示具体 PID、显存和 GPU util。是否继续由用户判断；代码不再因为还有大量空闲显存而强行阻止所有并行任务。

## 7. 训练输出

训练输出保存在：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/<experiment_group>/<run_id>/
```

典型文件：

```text
config.yaml                         运行用配置快照
config.full.yaml                    完整配置快照
dataset_statistics.json             normalization / unnormalization stats
summary.jsonl                       checkpoint 索引
topk_checkpoints.json               top-k checkpoint 排名
checkpoints/steps_<N>_pytorch_model.pt
accelerate_state/steps_<N>/         DeepSpeed/Accelerate 完整恢复状态
final_model/pytorch_model.pt
final_state/
```

`checkpoints/steps_<N>_pytorch_model.pt` 是权重；`accelerate_state/` 是继续训练需要的完整状态。

## 8. 评估

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

## 9. Ensemble

构建 CALVIN ultimate model soup：

```bash
bash interaction/bin/starvla-build-calvin-ultimate-ensemble.sh
```

脚本会读取 `interaction/core/ensemble.py` 中登记的成员 checkpoint，按权重构建 soup，并复制 config/statistics。生成的 ensemble checkpoint 可直接进入 websocket eval。

世界模型 Cosmos-Predict2-PI 当前可用权重整理脚本：

```bash
bash interaction/bin/starvla-copy-lzh-cosmos-final.sh
```

该脚本会把完整的 `steps_5500_pytorch_model.pt` 复制到 LZH 成员目录，同时创建标准 `final_model/pytorch_model.pt`。它会拒绝复制过小的疑似损坏 checkpoint。

## 10. 提交规范

提交前建议：

```bash
git status --short
python -m py_compile \
  deployment/model_server/policy_wrapper.py \
  deployment/model_server/policy_norm_processor.py \
  deployment/model_server/tools/websocket_policy_client.py \
  examples/LIBERO/eval_files/model2libero_interface.py \
  examples/calvin/eval_files/eval_calvin.py
```

当前分支目标名：

```text
star_vla_sii_summercamp
```

