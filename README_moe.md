# GTY MoE 预训练 / 后训练 / 自适应规划 — 执行说明

所有脚本位于 `public/seven/starvla_calvin/members/GTY/`。

## 前置条件

```bash
export P=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin
source $P/shared/runtime/starvla_env.sh
```

依赖路径（脚本已内置默认值，可按需覆盖）：

| 资产 | 路径 |
|------|------|
| Base VLM | `$P/shared/models/base/Qwen3-VL-4B-Instruct-Action` |
| CALVIN ABC 数据 | `$P/shared/datasets/calvin_lerobot` |
| CALVIN D 数据 | `$P/../inspire_shared/calvin_d_d` |
| CALVIN 评测 Python | `/inspire/.../public/four/miniconda3/envs/calvin_venv/bin/python` |

---

## 一、MoE 预训练

**脚本**：`members/GTY/run_train_abc_augmented_moe.sh`  
**模块**：`members/GTY/train_files/moe/moe_action_head.py`、`qwen_gr00t_moe.py`  
**框架**：`QwenGR00T_MoE`（4-expert soft-routing action decoder）

从头训练。冻结 Qwen3-VL-4B，训练 DiT-B + MoE Action Head。

```bash
# 3 GPU
NUM_PROCESSES=3 GPU_IDS=0,1,2 bash $P/members/GTY/run_train_abc_augmented_moe.sh

# 8 GPU
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 bash $P/members/GTY/run_train_abc_augmented_moe.sh

# 续训
IS_RESUME=true RUN_ID=abc_augmented_moe_GTY_0519_092147 \
  NUM_PROCESSES=3 GPU_IDS=0,1,2 bash $P/members/GTY/run_train_abc_augmented_moe.sh
```

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `NUM_PROCESSES` | 3 | 分布式进程数 |
| `GPU_IDS` | 0,1,2 | GPU 索引 |
| `MAX_TRAIN_STEPS` | 60000 | 总步数 |
| `SAVE_INTERVAL` | 1000 | checkpoint 间隔 |
| `BATCH_SIZE` | 16 | 单卡 batch |
| `IS_RESUME` | false | 续训开关 |

产出：
```
$P/members/GTY/runs/<RUN_ID>/
├── checkpoints/steps_<N>_pytorch_model.pt
├── final_model/pytorch_model.pt
└── training.log
```

---

## 二、MoE 后训练

**脚本**：`members/GTY/run_gty_moe_posttrain8.sh`  
**基座**：MoE 预训练 final_model（默认 60000 步）

```bash
# smoke 验证（20 步）
bash $P/members/GTY/run_gty_moe_posttrain8.sh smoke

# 夜间训练（8 GPU, 100000 步, 470 分钟 timeout）
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash $P/members/GTY/run_gty_moe_posttrain8.sh night

# 从其他 checkpoint 开始
PRETRAINED_CHECKPOINT=<path> \
  NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash $P/members/GTY/run_gty_moe_posttrain8.sh night
```

| 模式 | MAX_TRAIN_STEPS | SAVE_INTERVAL | TIMEOUT |
|------|----------------|---------------|---------|
| smoke | 20 | 20 | 无 |
| night | 100000 | 5000 | 470m |

日志：`$P/members/GTY/logs/<RUN_ID>.log`

---

## 三、MoE + 自适应规划

**脚本**：`members/GTY/run_train_abc_augmented_moe_adaptive.sh`  
**模块**：`members/GTY/train_files/adaptive/planner.py`、`moe/moe_adaptive_action_head.py`  
**框架**：`QwenGR00T_MoE_Adaptive`

训练与 MoE 完全相同，仅在推理时启用自适应早停。推理时每个 diffusion step 检测两个信号：

- **action delta** < 0.01 — 动作不再变化
- **MoE router entropy** < 0.5 — 专家达成一致

两步同时满足 + 跑够 min_steps（4 步）→ 提前退出，不跑满 max_steps（8 步）。

```bash
PRETRAINED_CHECKPOINT=<MoE 预训练 final_model 路径> \
  NUM_PROCESSES=3 GPU_IDS=0,1,2 \
  bash $P/members/GTY/run_train_abc_augmented_moe_adaptive.sh
```

YAML 配置（`starvla_calvin_abc_augmented_moe_adaptive.yaml`）：

```yaml
adaptive_planning:
  max_steps: 8
  min_steps: 4
  delta_threshold: 0.01
  entropy_threshold: 0.5
```

---

## 四、评测

**脚本**：`members/GTY/eval/run_eval_moe.sh`  
**Server**：`members/GTY/eval/run_policy_server_moe.sh`

CALVIN ABC→D 并行评测。每 GPU 启动一个 policy server，分片评测序列后自动汇总。

```bash
# 评测 MoE 预训练（默认 CKPT）
GPU_IDS=0,1,2,3 TOTAL_SEQUENCES=100 UNNORM_KEY=new_embodiment \
  bash $P/members/GTY/eval/run_eval_moe.sh

# 评测 MoE 后训练
CKPT=$P/members/GTY/runs/gty_moe_posttrain_8h_GTY_0519_182014/checkpoints/steps_95000_pytorch_model.pt \
  GPU_IDS=0,1,2,3 TOTAL_SEQUENCES=100 UNNORM_KEY=new_embodiment \
  bash $P/members/GTY/eval/run_eval_moe.sh
```

结果：`$P/members/GTY/reports/eval_moe_n<SEQ>_<timestamp>/results.json`

---

## 关键文件一览

```
members/GTY/
├── run_train_abc_augmented_moe.sh          # MoE 预训练
├── run_gty_moe_posttrain8.sh               # MoE 后训练
├── run_train_abc_augmented_moe_adaptive.sh # MoE + 自适应
├── eval/
│   ├── run_eval_moe.sh                     # 评测启动
│   └── run_policy_server_moe.sh            # Policy Server
├── train_files/
│   ├── moe/
│   │   ├── moe_action_head.py              # MoEActionDecoder + MoEFlowmatchingActionHead
│   │   ├── moe_adaptive_action_head.py     # MoEAdaptiveFlowmatchingActionHead
│   │   ├── qwen_gr00t_moe.py               # Framework: QwenGR00T_MoE
│   │   └── qwen_gr00t_moe_adaptive.py      # Framework: QwenGR00T_MoE_Adaptive
│   ├── adaptive/
│   │   └── planner.py                      # AdaptivePlanner
│   ├── data_registry/
│   │   └── data_config.py                  # GTYVideoColorJitter/RandomRotation
│   ├── run_train_moe_entry.py              # 训练入口 (MoE)
│   ├── run_train_moe_adaptive_entry.py     # 训练入口 (Adaptive)
│   ├── run_eval_server_entry.py            # 评测入口 (预导入框架)
│   └── starvla_calvin_abc_augmented_moe*.yaml
├── runs/                                    # 训练产出
│   ├── abc_augmented_moe_GTY_0519_092147/   # MoE 预训练 (60k 步, final_model 9.4G)
│   └── gty_moe_posttrain_8h_GTY_0519_182014/  # MoE 后训练 (95000 步, 无 final_model)
└── reports/                                 # 评测产出
    ├── eval_moe_n50_0519_154806/            # MoE 预训练 n=50
    └── eval_moe_n100_0520_*                 # MoE 后训练 n=100
```

## 当前结果

| 模型 | avg_seq_len | chain 1 | chain 2 | chain 3 | chain 4 | chain 5 |
|------|------------|---------|---------|---------|---------|---------|
| MoE 预训练 (n=50) | 1.54 | 78% | 40% | 24% | 8% | 4% |
| MoE 后训练 (n=100) | 1.91 | 76% | 54% | 32% | 17% | 12% |
