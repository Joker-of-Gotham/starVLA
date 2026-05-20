# StarVLA SII SummerCamp 使用与维护说明

本文档面向本分支的训练、后训练、评估、ensemble 和仓库维护。顶层 README 保留上游 StarVLA 项目介绍；本文件解释当前分支新增的工程化能力、目录规范和推荐操作方式。技术路线和失败模式分析见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)，CALVIN ABC→D 完整测评表见 [CALVIN_EVALUATION_REPORT.md](CALVIN_EVALUATION_REPORT.md)，总入口见 [SUMMERCAMP_REPORT.md](SUMMERCAMP_REPORT.md)。

## 0. 从离线环境到正式评测的端到端路线

如果是新机器、新容器或重新拉取代码，按下面顺序做。每一步都能独立检查，出错时优先停在当前步骤修正，不要直接进入长时间训练。

| 步骤 | 目标 | 命令或入口 | 成功标准 |
|---|---|---|---|
| 1 | 进入仓库 | `cd /inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA` | 当前目录包含 `starVLA/`、`interaction/`、`examples/` |
| 2 | 安装或恢复离线环境 | `bash envSet/starvla_env/scripts/install_offline.sh` | `env/starvla-py310` 存在，关键包导入通过 |
| 3 | 激活环境 | `source envSet/starvla_env/scripts/activate_offline_env.sh` | `which python` 指向 `env/starvla-py310/bin/python` |
| 4 | 完整环境自检 | `bash envSet/starvla_env/scripts/check_offline_env.sh` | Python、CUDA、DeepSpeed、transformers、tmux、NCCL 工具均通过 |
| 5 | GPU/NCCL 快速修复 | `bash interaction/bin/starvla-gpu-fix.sh --num-gpus 8` | 缺失的 editable install、flash-attn、transformers overlay 被修正 |
| 6 | 查看目录和组合空间 | `bash interaction/bin/starvla-interact.sh paths`、`catalog --kind all` | 数据集、base model、action expert、T/S policy 都可见 |
| 7 | 检查 H200 通信和负载 | `bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8` | 看到 H200、NVLink/NCCL profile、all_reduce_perf/p2p 工具状态 |
| 8 | 训练 dry-run | `bash interaction/bin/starvla-interact.sh train ... --dry-run` | 命令、输出目录、action/state/horizon、checkpoint 路径正确 |
| 9 | 正式训练 | 交互主菜单选 `launch_train`，或 `train --yes` | tmux job 建立，`monitor` 有 step/loss/GPU 心跳 |
| 10 | 续训或磁盘守护 | `run-mode continue` 或 `starvla-disk-guard-resume.sh` | 从最新 `accelerate_state` 或权重恢复，不重复新建错误 run |
| 11 | 后训练/ensemble | `--posttrain-from` 或 `ensemble --recipe ...` | 新 run 从已有权重初始化，或产出 `steps_ensemble_pytorch_model.pt` |
| 12 | 评估与汇总 | `eval --preset calvin --ckpt ...` | server/client 日志正常，`merged_results.json` 产出 |

推荐先跑一次最小闭环：

```bash
source envSet/starvla_env/scripts/activate_offline_env.sh
bash interaction/bin/starvla-interact.sh check
bash interaction/bin/starvla-interact.sh catalog --kind all
bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --training-policy T01 T06 T07 \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --structure-policy S01 S04 S07 S15 S27 \
  --gpus auto --num-gpus 8 \
  --throughput-profile auto \
  --dry-run
```

确认 dry-run 打印的 `output_dir`、`checkpoint_dir`、`run_id`、`framework.action_model.action_dim/state_dim/action_horizon` 都正确后，再去掉 `--dry-run`。

## 0.1 离线环境配置

离线包体积较大，默认不进入 Git。当前机器上建议把团队提供的离线包放在：

```text
starVLA/envSet/starvla_env/
  archives/starvla-py310-env.tar
  repo_overlay/
  scripts/install_offline.sh
  scripts/activate_offline_env.sh
  scripts/check_offline_env.sh
  manifest.json
```

如果新 clone 的仓库没有 `envSet/starvla_env/`，先从公共盘或成员机器同步该目录，再执行安装。同步后可先校验归档：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA/envSet/starvla_env
sha256sum -c archives/starvla-py310-env.tar.sha256 2>/dev/null || true
```

如果没有 `.sha256` 文件，则以 `manifest.json` 中记录的归档大小、路径和版本信息为准。

第一次安装：

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/26220447/starVLA
bash envSet/starvla_env/scripts/install_offline.sh
source envSet/starvla_env/scripts/activate_offline_env.sh
bash envSet/starvla_env/scripts/check_offline_env.sh
```

常用安装参数：

| 参数 | 含义 | 什么时候用 |
|---|---|---|
| `--force-restore` | 删除并重新解压 `env/starvla-py310` | 环境损坏、包版本混乱、Python 导入异常 |
| `--force-reinstall` | 在已有环境中强制重装 overlay/关键 wheel | editable install 或 transformers/flash-attn 覆盖异常 |
| `--repo-root PATH` | 指定 StarVLA 仓库根目录 | 脚本不在仓库根目录执行时 |
| `--env-dir PATH` | 指定 Python 环境目录 | 希望环境放到非默认位置时 |
| `--skip-interaction-overlay` | 不覆盖 interaction 文件 | 调试本地 interaction 修改时 |
| `--skip-check` | 安装后不跑检查 | 只想快速恢复环境，稍后手动检查时 |

激活脚本会做三件事：

- 激活 `env/starvla-py310`。
- 设置 `STARVLA_REPO_ROOT`、`STARVLA_ENVSET_ROOT`、`STARVLA_ENV_DIR`。
- 清理代理变量并设置离线路径，避免 HuggingFace/ModelScope 在训练中意外联网卡住。

环境检查脚本会检查：

- `tmux`、`all_reduce_perf`、`p2pBandwidthLatencyTest` 是否存在。
- `torch`、`accelerate`、`deepspeed`、`transformers`、`flash_attn`、`diffusers`、`qwen_vl_utils` 是否能导入。
- CUDA 是否可用、GPU 数量是否满足 `--num-gpus`。
- `interaction/bin/starvla-interact.sh check` 和 `perf-check` 是否能跑通。

如果 H200 节点上出现导入、CUDA、NCCL 或 editable install 问题，优先使用：

```bash
bash interaction/bin/starvla-gpu-fix.sh --num-gpus 8
```

它是幂等修复脚本：健康环境会快速跳过，只有缺失或版本不一致时才修复。确认环境后再运行：

```bash
python - <<'PY'
import torch, transformers, accelerate, deepspeed, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("gpus", torch.cuda.device_count())
PY
```

期望看到 CUDA 可用、GPU 数量正确，并且 `transformers` 是离线包 overlay 后支持 Qwen3.5/Qwen3-VL 的版本。

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

也可以直接用子命令，适合写脚本：

```bash
bash interaction/bin/starvla-interact.sh check
bash interaction/bin/starvla-interact.sh catalog --kind all
bash interaction/bin/starvla-interact.sh perf-check --gpus auto --num-gpus 8
bash interaction/bin/starvla-interact.sh train --dataset calvin_abc --dry-run
bash interaction/bin/starvla-interact.sh eval --preset calvin --ckpt /path/to/pytorch_model.pt
```

### 6.1 主菜单每个选项的含义

| 菜单项 | 作用 | 什么时候选 | 主要输出 |
|---|---|---|---|
| `check` | 检查环境、路径、Python 导入、action head 注册 | 新节点、新环境、训练/评估前 | 表格显示 OK/FAIL 和具体路径 |
| `catalog` | 展示 datasets、base models、action experts、Training/Structure Policy | 不确定有哪些可选组合时 | 全部可选 key、说明和推荐策略 |
| `perf_check` | 检查 GPU 拓扑、H200/NCCL profile、带宽测试工具 | 训练前或怀疑多卡通信慢时 | GPU、NVLink、NCCL、all_reduce_perf、p2p 工具状态 |
| `paths` | 展示 checkpoint、run、tmux 产物保存路径 | 找不到输出目录或准备复制结果时 | 默认 checkpoint root 和 run root |
| `selftest` | 启动一个 CPU-only tmux 小测试 | 验证 tmux 管理层是否正常 | 测试 job、日志、事件文件 |
| `ensemble` | 多个 checkpoint 做 model soup | 正式评估前想融合多个强模型，或后训练前初始化 | ensemble run 目录和 `steps_ensemble_pytorch_model.pt` |
| `launch_train` | 交互式启动训练/续训/后训练 | 日常训练推荐使用 | 确认表、tmux session、日志路径 |
| `launch_eval` | 交互式启动 websocket 评估 | CALVIN/LIBERO 等正式或快速测评 | policy server/client tmux 窗口、评估结果 |
| `list` | 列出受管理 job | 不知道当前有哪些训练/评估在跑时 | job 名、状态、路径 |
| `monitor` | 原位置刷新训练/评估监控面板 | 观察 loss、progress、GPU、错误根因 | Rich/ANSI dashboard，不持续刷屏 |
| `attach` | 进入 tmux session | 需要看完整实时日志或手动调试时 | 进入对应 tmux 窗口 |
| `stop` | 停止 tmux job | 主动终止训练/评估或清理卡住 job | 发送 TERM/KILL 并更新事件 |
| `exit` | 退出交互菜单 | 不再操作 | 返回 shell |

### 6.2 `launch_train` 每一步怎么选

交互式训练会按下面顺序询问：

1. `Dataset`：选择训练数据集。它决定 `data_root`、`data_mix`、`action_dim`、`state_dim`、`action_horizon`、normalization group。
2. `Action expert`：选择动作专家。抽象项如 `OFT/PI/GR00T/FAST` 会按 base model family 自动解析到具体 framework；具体项如 `QwenOFT/QwenFast/CosmoPredict2PI` 会直接使用。
3. `Base model`：选择 VLM 或世界模型 backbone。注意 action-token 路线优先选带 `_action` 的模型。
4. `Training policies`：输入 `T01 T06 T07` 这类列表，逗号或空格均可。留空会使用该 dataset 的推荐前三项。
5. `Structure policies`：输入 `S01 S04 S07` 这类列表。留空会使用该 dataset 的推荐前三项。
6. `Run mode`：`new` 新建 run；`continue` 续训同一个 run。
7. `Resume mode`：仅 `continue` 时出现。`auto` 优先完整 state，失败时回退权重；`full` 要求 optimizer/scheduler/RNG；`weights` 只加载模型权重。
8. `Post-train from checkpoint`：仅 `new` 时出现。填已有 `pytorch_model.pt` 表示基于该模型另开后训练 run；留空表示从 base pretrained 开始。
9. `Reload modules from checkpoint`：后训练时可选。留空加载完整模型；也可填 `action_model,qwen_vl_interface` 等模块名做局部加载。
10. `GPUs`：`auto` 自动选可用卡；`all` 使用可见所有卡；`0,1,2,3` 指定卡。共享节点建议手动指定。
11. `Number of GPUs when auto`：`auto/all` 时最多选几张卡。8 卡 H200 正式训练填 `all` 或 `8`。
12. `run_id`：留空自动生成。续训时留空会使用当前组合下最新 run；也可填已有 run_id。
13. `max_train_steps`：留空使用 preset；调试可填小值，例如 `100` 或 `1000`。
14. `Throughput profile`：控制 batch/dataloader。`auto` 会在 H200 上解析到 aggressive；`h200_saturated` 强制大 batch；`balanced` 中等；`conservative` 保守；`none` 不覆盖 YAML。
15. `VLA per-device batch`：每张 GPU 的动作训练 batch。显存充足可显式填大值，显式值不会被自动降级。
16. `VLM per-device batch`：co-train/VLM-only 分支的语言视觉 batch。普通 VLA 训练通常影响较小。
17. `DataLoader workers per rank`：每个 GPU 进程的 dataloader worker 数。H200 通常从 `8` 开始，IO 弱时降到 `4`。
18. `Confirm launch`：输入 `yes` 或 `y` 才会真正创建 tmux job。

`new`、`continue`、`post-train` 不要混用：

- 从 base pretrained 开始：`Run mode = new`，`Post-train from checkpoint` 留空。
- 继续同一个中断 run：`Run mode = continue`，`Resume mode = auto`。
- 基于已有训练好模型做后训练：`Run mode = new`，填写 `Post-train from checkpoint`。

### 6.3 数据集、模型、动作专家选择建议

| 选择类型 | 推荐选择 | 适用情况 |
|---|---|---|
| CALVIN 主线 | `dataset=calvin_abc` | 当前最常用训练和评估闭环 |
| CALVIN 兼容/迁移 | `dataset=calvin_abc_d` | 需要 ABC-D 或 D-D 兼容数据时 |
| OXE/Bridge | `oxe_bridge`、`oxe_bridge_rt1` | 跨机器人预训练、mixture 训练 |
| RoboTwin | `robotwin_all`、`robotwin_all_50` | 双臂、长 horizon、14D 动作 |
| RoboCasa GR1 | `robocasa_gr1` | humanoid/GR1 29D 高维动作 |
| VLM-only | `vlm_robot` | 只做图像语言预/后训练，不输出动作 |
| 稳定 baseline | `QwenOFT` 或抽象 `OFT` | continuous BC，最适合先建立可跑 baseline |
| flow 路线 | `QwenPI`、`QwenGR00T`、抽象 `PI/GR00T` | 多模态连续动作、高维或长 horizon |
| token 路线 | `QwenFast`、抽象 `FAST` | action-token/DCT/FAST，base model 应选 `_action` |
| 世界模型路线 | `cosmos_predict2 + PI/OFT/GR00T` 或 `CosmoPredict2*` | 尝试 Cosmos Video2World 作为 action 条件/辅助 |
| 快速小模型 | `qwen35_0_8b`、`qwen35_2b` | 小 epoch、小 batch、快速验证 |
| 当前强视觉主线 | `qwen3vl_4b`、`qwen3vl_4b_action` | CALVIN/FAST/OFT/PI 主实验 |

### 6.4 常用 Training Policy 组合

| 目标 | 推荐 Training Policy | 说明 |
|---|---|---|
| OFT/MLP baseline | `T01 T02 T06 T07 T11 T22` | normalization、mixture、continuous BC、chunk curriculum、gripper/group、增强 |
| FAST/action-token | `T01 T02 T04 T05 T07 T11 T22` | action token embedding warmup、DCT/FAST tokenizer、chunk curriculum |
| PI/GR00T flow | `T01 T02 T07 T08 T11 T22` | flow matching action expert、chunk、group/gripper |
| diffusion/DiT | `T01 T02 T07 T09 T11 T22` | 高维动作或多模态动作分布 |
| VLA+VLM co-train | `T01 T02 T12 T13 T22` | 动作 loss 加 VLM/grounding/future/inverse 辅助 |
| 世界模型辅助 | `T12 T15 T16 T23` | future/world latent 训练期正则，推理期可关闭 |
| 后训练 | `T20 T21 T24 T27 T28` | hard-task replay、anti-forgetting、offline RL、DPO、EMA/SWA |

### 6.5 常用 Structure Policy 组合

| 目标 | 推荐 Structure Policy | 说明 |
|---|---|---|
| OFT baseline | `S01 S04 S07 S15 S27` | base preservation、embodiment/FPS token、MLP head、gripper、安全边界 |
| FAST/action-token | `S01 S02 S04 S08 S15 S27` | action token embedding、FAST token head、gripper、安全边界 |
| PI/GR00T flow | `S01 S04 S10 S11 S15 S27` | flow expert、layerwise fusion、gripper、安全边界 |
| high-DoF/多相机 | `S03 S05 S06 S13 S14 S24` | multi-camera、proprio、temporal memory、bimanual/GR1 decoder、3D connector |
| MoE/domain adapter | `S16 S17 S30` | embodiment/task/domain routing，适合多模型/多域融合 |
| world/future | `S20 S21 S22 S31` | latent intent、world head、future representation、Cosmos teacher |
| 评估安全 | `S26 S27 S28` | value/success、action-bound、安全/OOD 不确定性 |

### 6.6 训练常用命令

```bash
# CALVIN OFT baseline，从 base pretrained 开始
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --training-policy T01 T02 T06 T07 T11 T22 \
  --structure-policy S01 S04 S07 S15 S27 \
  --gpus auto --num-gpus 8 \
  --throughput-profile auto

# FAST/action-token 训练
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenFast \
  --training-policy T01 T02 T04 T05 T07 T11 T22 \
  --structure-policy S01 S02 S04 S08 S15 S27 \
  --gpus auto --num-gpus 8 \
  --throughput-profile h200_saturated

# 续训最新 run
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --run-mode continue \
  --resume-mode auto \
  --gpus auto --num-gpus 8

# 基于已有权重做后训练
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --training-policy T20 T21 T24 T27 T28 \
  --structure-policy S01 S26 S27 S30 \
  --posttrain-from /path/to/checkpoints/steps_60000_pytorch_model.pt \
  --gpus auto --num-gpus 8
```

## 7. 自由组合流程

本分支把训练配置拆成五层：

```text
datasets -> training policy -> base model -> action expert -> structure policy
```

完整的 Training Policy 表 1 和 Structure Policy 表 2 已整理到 [POLICY_MATRIX.md](POLICY_MATRIX.md)。该矩阵是交互式输入的规范解释：`Txx` 说明训练阶段、训练哪些参数、loss/采样/后训练策略；`Sxx` 说明结构放在哪里、改哪个模块、与 action expert 的关系。日常使用可以先看本节的组合规则，遇到不确定的 policy 再查完整矩阵。

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

完整矩阵的使用规则：

- 从头训练先固定 `T01 T02 T07` 和 `S01 S04 S27`，再按 action expert 添加专用策略。
- OFT 路线补 `T06 T11 T22` 和 `S07 S15`。
- FAST/action-token 路线补 `T04 T05 T11 T22` 和 `S02 S08 S15`，base model 优先选带 `_action` 的版本。
- PI/GR00T flow 路线补 `T08 T11 T22` 和 `S10 S11 S15`。
- 世界模型/未来表征路线补 `T15 T16 T23` 和 `S20 S21 S22 S31`，适合 `cosmos_predict2` 或 `CosmoPredict2*`。
- 后训练另开新 run，使用 `--posttrain-from`，通常选 `T20 T21 T24 T27 T28` 和 `S26 S27 S28 S30`。
- `launch-supported` 表示已有原生入口或原生 framework/head；`runtime-supported` 表示通过共享 trainer runtime hook、metadata、regularizer、loss weight、deployment metadata 或后训练流程落地，能随 checkpoint 续训或后训练运行。

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

CALVIN websocket eval 可以走交互菜单 `launch_eval`，也可以直接用子命令。

```bash
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu all \
  --trials 1000
```

### 11.1 `launch_eval` 每一步怎么选

1. `Eval preset`：选择 benchmark。当前包含 `libero`、`calvin`、`robotwin`、`robocasa365`；正式 CALVIN 选 `calvin`。
2. `checkpoint path`：填模型权重文件，例如 `checkpoints/steps_60000_pytorch_model.pt`、`final_model/pytorch_model.pt` 或 ensemble 的 `steps_ensemble_pytorch_model.pt`。
3. `Evaluation Scale`：CALVIN 专用。
   - `quick`：20 sequences、每个 subtask 最多 120 steps，适合 smoke test。
   - `medium`：100 sequences、360 steps，适合 10 到 30 分钟级别比较。
   - `official`：1000 sequences、360 steps，正式分数。
   - `custom`：手动输入 sequence 数和 max steps。
4. `server GPU(s)`：policy server 放在哪些 GPU。`auto` 自动选，`all` 使用可见所有卡，`0,1,2,3` 手动指定。多卡评估时优先让 server 分布到多张卡。
5. `Eval Throughput Profile`：评估并发策略。
   - `auto`：H200 上自动使用多 server replica 和共享 client。
   - `h200_saturated`：更激进，适合显存和 CPU 都充足时。
   - `balanced`：每卡多个 server/client 的中等配置。
   - `conservative`：每卡一个 server replica，最稳。
6. `Policy server replicas per GPU`：每张 GPU 加载几个模型副本。模型小、显存空闲多时可填 `2` 或更高；模型大或 OOM 时填 `1`。
7. `CALVIN clients per policy server`：每个 server 挂几个环境 client。CPU/渲染资源充足时可提高；如果卡顿、环境启动慢或端口异常，降到 `1`。
8. `Concurrent inference requests per server`：单个 server 同时处理多少请求。通常 `1-2` 比较稳，过高会造成排队和显存峰值。
9. `Total eval workers`：总 rollout worker 数。留空让 profile 自动计算；正式测评前建议先 medium 验证。
10. `CALVIN Render Backend`：
    - `auto`：优先 EGL GPU headless，失败后回退。
    - `egl`：强制 GPU headless 渲染，速度快但环境必须支持。
    - `direct`：安全 CPU/software fallback，慢但更稳。
11. `Confirm launch`：确认后会创建 tmux session，包含 server/client 多个窗口。

### 11.2 CALVIN 推荐评估路线

```bash
# 1. 先 smoke test，确认模型能加载、server/client 能通信
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu 0 \
  --trials 20 \
  --max-steps-per-task 120 \
  --eval-throughput-profile conservative

# 2. 中等规模比较，用于筛 checkpoint 或 ensemble 权重
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu 0,1,2,3 \
  --trials 100 \
  --max-steps-per-task 360 \
  --eval-throughput-profile balanced

# 3. 正式 1000 sequences
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu all \
  --trials 1000 \
  --max-steps-per-task 360 \
  --eval-throughput-profile auto
```

如果显存使用率很低但速度慢，优先增加 `--replicas-per-gpu` 或 `--clients-per-server`，而不是只增加 GPU 数。例如：

```bash
bash interaction/bin/starvla-interact.sh eval \
  --preset calvin \
  --ckpt /path/to/pytorch_model.pt \
  --server-gpu 0,1,2,3 \
  --trials 1000 \
  --replicas-per-gpu 2 \
  --clients-per-server 2 \
  --server-concurrency 2 \
  --eval-throughput-profile h200_saturated
```

如果 CPU、渲染或端口成为瓶颈，则反向降低：

```bash
--replicas-per-gpu 1 --clients-per-server 1 --server-concurrency 1 --eval-throughput-profile conservative
```

### 11.3 评估输出在哪里看

交互式评估会把运行产物放在：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/<timestamp>_starvla_eval_<name>/
  job.json
  logs/
    server*.log
    client*.log
  calvin_eval/
    workers/
    merged_results.json
    sequence_progress.json
```

重点文件：

- `logs/server*.log`：模型加载、显存、policy inference、normalization key、WebSocket 端口。
- `logs/client*.log`：CALVIN 环境初始化、渲染 backend、每条 sequence rollout。
- `calvin_eval/workers/*/sequence_progress.json`：worker 粒度进度。
- `calvin_eval/merged_results.json`：合并后的 success、avg length、各 subtask 统计。
- `job.json`：本次评估的 checkpoint、GPU、端口、并发参数。

本分支修复了一个常见问题：client 请求 `unnorm_key=franka`，但 checkpoint 只有 `new_embodiment`。服务端现在会在单 key checkpoint 下自动映射到唯一可用 key；客户端也会根据 server metadata 做同样修正。

如果评估失败，优先检查：

- `logs/server*.log`：模型加载、OOM、norm key、policy inference error
- `logs/client*.log`：环境初始化、websocket 响应、rollout 错误
- `calvin_eval/workers/*/sequence_progress.json`：每个 worker 的当前进度
- `calvin_eval/merged_results.json`：合并后的最终统计

常见问题和处理：

| 现象 | 先看哪里 | 处理方式 |
|---|---|---|
| server 很快退出 | `logs/server*.log` | 检查 checkpoint 路径、framework、dataset_statistics、显存 OOM |
| client 卡住不动 | `logs/client*.log` | 降低 clients/server，换 `--calvin-render-backend direct` 验证 |
| 端口冲突 | `job.json`、server log | 指定 `--port` 或停止旧 eval job |
| GPU 显存空但速度慢 | monitor GPU/CPU 心跳 | 增加 replicas/client 并发；确认不是渲染或 CPU worker 瓶颈 |
| progress 总数不对 | `job.json` 和 `sequence_progress.json` | 检查 `--trials`、worker 数、是否使用 official/custom scale |
| `NoneType tokens`/decode 错误 | server log | 多见于 action-token 路线输出为空，检查 base model 是否为 `_action`、FAST policy 是否包含 `T04/T05` |
| state/action 维度错误 | server log | 检查 dataset key、`dataset_statistics.json`、`framework.action_model.*` 是否匹配 |

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
