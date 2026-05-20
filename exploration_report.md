# GTY starVLA 探索报告

> 项目路径：`members/GTY/` @ `starVLA_calvin`
> 时间跨度：2026-05-19 ~ 2026-05-20
> 目标：在 CALVIN ABC→D 任务上探索数据增强、MoE 动作头、自适应规划等方案

---

## 一、基座模型

| 组件 | 规格 |
|------|------|
| VLM | Qwen3-VL-4B-Instruct-Action（frozen） |
| Action Model | DiT-B（trainable） |
| Action Head | Flow Matching，action_dim=7（6-DoF + gripper），horizon=16 |
| 训练框架 | DeepSpeed Zero-2 + Accelerate |
| 硬件 | 3~8 × NVIDIA H200（141GB/卡） |
| 数据 | CALVIN ABC（LeRobot v2.0 格式） |

---

## 二、探索路径总览

```
Phase 1: Head-only Baseline
    │
Phase 2: Data Augmentation (ColorJitter + RandomRotation)
    │
Phase 3: MoE Action Head (4 experts, soft routing)
    │
    ├── Phase 4: MoE Post-train (续训 100k steps)
    │       │
    │       └── Phase 6: WMH LoRA (基于 post-train 95000)
    │
    └── Phase 5: MoE + Adaptive Planning (收敛早停)
```

---

## 三、各阶段详情

### Phase 1 — Head-only Baseline

**目标**：仅训练 action head，冻结 VLM，建立 baseline。

**新增文件**：无（使用共享管线）

| Run ID | 状态 |
|--------|------|
| `abc_headonly_GTY_0519_053442` | 早期尝试 |
| `abc_headonly_GTY_0519_053739` | 调试 |
| `abc_headonly_GTY_0519_065811` | 调试 |
| `abc_headonly_GTY_0519_071249` | 调试 |

> 此阶段主要验证训练流程可行性，未保留最终模型。

---

### Phase 2 — 数据增强

**目标**：训练时对视频帧施加 ColorJitter + RandomRotation，提升泛化能力。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `train_files/data_registry/data_config.py` | `GTYVideoColorJitter`、`GTYVideoRandomRotation`（容忍 eval 时无 video metadata） |
| `train_files/starvla_calvin_abc_augmented.yaml` | 增强训练配置 |
| `run_train_abc_augmented.sh` | 启动脚本 |

**关键 Bug 修复**：
- `VideoToTensor.__array_interface__` 类型错误
- `GTYVideoColorJitter.set_metadata()` 在 eval 环境下因缺少 video metadata 而崩溃 → 改为空安全跳过

**Runs**：

| Run ID | 结果 |
|--------|------|
| `abc_augmented_GTY_0519_065016` | 调试 |
| `abc_augmented_GTY_0519_065124` | 调试 |
| `abc_augmented_GTY_0519_070012` | 调试 |
| `abc_augmented_GTY_0519_071410` | 调试 |
| `abc_augmented_GTY_0519_071546` | 调试 |
| `abc_augmented_GTY_0519_071915` | 调试 |
| `abc_augmented_GTY_0519_072827` | 最终版 |

---

### Phase 3 — MoE 动作头

**目标**：将原始单一 MLP action decoder 替换为 Mixture of Experts，提升多模态动作建模能力。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `train_files/moe/moe_action_head.py` | `MoEActionDecoder`（4 experts, soft routing）+ `MoEFlowmatchingActionHead` |
| `train_files/moe/qwen_gr00t_moe.py` | `@FRAMEWORK_REGISTRY.register("QwenGR00T_MoE")` |
| `train_files/starvla_calvin_abc_augmented_moe.yaml` | MoE 训练配置 |
| `train_files/run_train_moe_entry.py` | 预导入 MoE 后委托 train_starvla.main() |
| `run_train_abc_augmented_moe.sh` | 启动脚本（resume + tee 日志） |

**MoE 结构**：

```
DiT output [B, T, 1024]
      │
      ▼
Router: Linear(1024 → 4) → softmax → [w0, w1, w2, w3]
      │
      ├── Expert 0: Linear(1024→256) → ReLU → Linear(256→7)
      ├── Expert 1: Linear(1024→256) → ReLU → Linear(256→7)
      ├── Expert 2: Linear(1024→256) → ReLU → Linear(256→7)
      └── Expert 3: Linear(1024→256) → ReLU → Linear(256→7)
      │
      ▼
Σ w_i × expert_i(x)  →  [B, T, 7]
```

**Runs**：

| Run ID | 结果 |
|--------|------|
| `abc_augmented_moe_GTY_0519_083432` | 调试 |
| `abc_augmented_moe_GTY_0519_085526` | 调试 |
| `abc_augmented_moe_GTY_0519_092147` | ✅ **成功**，60000 步，final_model 9.4G |

**CALVIN ABC→D 评测结果（n=50）**：

```
avg_seq_len: 1.54
chain 1: 78%  ████████
chain 2: 40%  ████
chain 3: 24%  ██
chain 4:  8%  █
chain 5:  4%
```

---

### Phase 4 — MoE 后训练

**目标**：在 MoE 预训练基础上继续训练 100000 步，进一步提升性能。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `run_gty_moe_posttrain8.sh` | 后训练启动脚本（smoke/night 模式，timeout 控制） |

**Runs**：

| Run ID | 结果 |
|--------|------|
| `gty_moe_posttrain_smoke20_GTY_0519_163613` | smoke 测试通过 |
| `gty_moe_posttrain_8h_GTY_0519_164514` | ❌ 磁盘不足（35GB < 50GB），被 watchdog 拦截 |
| `gty_moe_posttrain_8h_GTY_0519_182014` | ⚠️ 95000/100000 步，被 `timeout 470m` 杀掉 |

> 最新可用 checkpoint：`steps_95000_pytorch_model.pt`，无 final_model。

---

### Phase 5 — 自适应规划

**目标**：在 diffusion 推理时根据收敛状态自适应早停，减少冗余计算。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `train_files/adaptive/planner.py` | `AdaptivePlanner`：双信号收敛判断（action delta + MoE router entropy） |
| `train_files/moe/moe_adaptive_action_head.py` | `MoEAdaptiveFlowmatchingActionHead`：仅覆写 `predict_action()` |
| `train_files/moe/qwen_gr00t_moe_adaptive.py` | `QwenGR00T_MoE_Adaptive` 框架注册 |
| `train_files/starvla_calvin_abc_augmented_moe_adaptive.yaml` | 配置（max_steps=8, min_steps=4） |
| `train_files/run_train_moe_adaptive_entry.py` | entry point |
| `run_train_abc_augmented_moe_adaptive.sh` | 启动脚本 |

**自适应规划逻辑**：

```
每个 denoising step:
  1. DiT 去噪 → MoE 预测 action
  2. 计算 delta  = |action_new - action_old| / |action_old|
  3. 计算 entropy = -Σ p_i·log(p_i)  (MoE router)
  4. if step >= min_steps AND delta < 阈值 AND entropy < 阈值:
       → 提前退出
  5. else → 继续
```

**Run**：

| Run ID | 结果 |
|--------|------|
| `abc_augmented_moe_adaptive_WMH_0519_112344` | ⚠️ 仅跑了一次，命名误用 WMH，未充分验证 |

---

### Phase 6 — 评测管线

**目标**：构建 GTY 专用的可靠评测管线，解决 MoE 框架注册等问题。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `train_files/run_eval_server_entry.py` | 预导入 QwenGR00T_MoE 后启动 policy server |
| `eval/run_policy_server_moe.sh` | GTY MoE 专用 server 启动器 |
| `eval/run_eval_moe.sh` | 完整并行评测脚本（多 GPU + 自动汇总） |

**关键 Bug 修复**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Framework QwenGR00T_MoE not implemented` | eval 环境未导入 MoE 模块 | `run_eval_server_entry.py` 预导入 |
| `Video key primary_image not found` | eval 环境无 video metadata | `GTYVideoColorJitter.set_metadata()` 空安全 |
| Shell 语法错误导致 CKPT 未生效 | `env CKPT=...` 缺少 `\` 续行 | 使用 `export` 或 `\` 续行 |

---

## 四、辅助工具

| 文件 | 说明 |
|------|------|
| `tools/watchdog/train_watchdog.sh` | 训练守护进程（检查磁盘/进程状态，自动 resume） |
| `tools/watchdog/watch_status.sh` | 批量监控多个 job |

---

## 五、磁盘使用

GTY runs 目录共 **27 个子目录**，每个 checkpoint 约 10GB。早期调试 runs 可清理以释放空间。

---

## 六、当前状态 & 待办

| 优先级 | 事项 | 状态 |
|--------|------|------|
| **P0** | MoE 后训练评测（step 95000） | 权重就绪，可立即评测 |
| **P1** | MoE 后训练 resume 到 100000 步 | 95000 checkpoint 就绪，需延长 timeout |
| **P2** | MoE+Adaptive 用 GTY 权重重新训练 | 当前只跑过 WMH 命名的一次 |
| **P2** | 各阶段 CALVIN ABC→D 评测对比 | 目前仅 MoE pretrain 有 n=50 结果 |
| **P3** | 清理早期失败 runs 释放磁盘 | 27 个 runs，可保留 3-5 个关键版本 |

---

## 七、评测方法

```bash
# MoE 预训练模型
CKPT=.../abc_augmented_moe_GTY_0519_092147/final_model/pytorch_model.pt \
GPU_IDS=0,1,2,3 TOTAL_SEQUENCES=1000 \
bash members/GTY/eval/run_eval_moe.sh

# MoE 后训练模型（以 step 95000 为例）
CKPT=.../gty_moe_posttrain_8h_GTY_0519_182014/checkpoints/steps_95000_pytorch_model.pt \
GPU_IDS=0,1,2,3 TOTAL_SEQUENCES=1000 \
bash members/GTY/eval/run_eval_moe.sh
```

结果输出到 `members/GTY/reports/eval_moe_n<N>_<timestamp>/results.json`，包含 `avg_seq_len` 和 chain 1~5 成功率。
