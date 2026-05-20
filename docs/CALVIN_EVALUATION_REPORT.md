# StarVLA CALVIN ABC→D 评测与结果分析报告

本文面向 StarVLA 在 CALVIN ABC→D 长程任务上的测评结果、指标解释、结果归因和个人贡献说明。技术路线、训练策略、模型结构与 failure pattern 的完整理论分析见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)；工程运行、交互式训练与评估流程见 [SUMMERCAMP_STARVLA.md](SUMMERCAMP_STARVLA.md)。

## 1. 报告摘要

CALVIN ABC→D 测评用于检验策略从训练环境 ABC 迁移到未见环境 D 的长程泛化能力。每条 evaluation sequence 由最多 5 个 subtasks 组成，结果以 `Task 1` 到 `Task 5` 的连续成功率和 `Average Chain Length` 衡量。该指标不仅反映单步 affordance 和动作精度，也显式暴露任务切换、状态恢复、接触时序和误差累积问题。

当前可解析结果中，`n=300` 规模下 **WMH MoE95k LoRA Aug latest** 达到 Avg Len `1.863`、Task 5 `12.7%`；`n=1000` 规模下 **WMH state8 connector steps8000 fast w4x8** 达到 Avg Len `1.086`、Task 5 `4.3%`。对比结果表明，MoE、LoRA 后训练和 augmentation 的收益主要体现在 Task 3~5 的长链保持能力，而不是只提升第一步成功率。参数级 checkpoint soup ensemble 表现明显退化，说明异构 MoE/router/action head 直接平均会破坏专家分工和动作模式。

## 2. 测评任务与指标定义

### 2.1 CALVIN ABC→D 协议

CALVIN ABC→D 要求策略只在 ABC 环境或 ABC 相关数据上训练，然后在未见过的 D 环境中执行多步语言指令链。每条 sequence 最多包含 5 个 subtasks，因此测评结果包含：

- `Task 1`：至少完成第 1 个 subtask 的 sequence 比例。
- `Task 2`：连续完成前 2 个 subtasks 的 sequence 比例。
- `Task 3`：连续完成前 3 个 subtasks 的 sequence 比例。
- `Task 4`：连续完成前 4 个 subtasks 的 sequence 比例。
- `Task 5`：连续完成全部 5 个 subtasks 的 sequence 比例。
- `Average Chain Length`：每条 sequence 平均完成的 subtask 数。

该协议的关键不是孤立的单步 success，而是 survival curve：策略必须在完成前一任务后，从新状态继续理解下一条指令并执行正确动作。

### 2.2 数学定义

令第 $i$ 条 sequence 完成长度为 $l_i\in\{0,1,2,3,4,5\}$，总评估条数为 $N$。第 $k$ 个位置的 survival success rate 为：

$$
\mathrm{SR}_k
=\frac{1}{N}\sum_{i=1}^{N}\mathbb 1[l_i\ge k],
\qquad k=1,\ldots,5.
$$

平均链长为：

$$
\mathrm{AvgLen}
=\frac{1}{N}\sum_{i=1}^{N}l_i
=\sum_{k=1}^{5}\mathrm{SR}_k.
$$

因此 `Average Chain Length` 与 Task 1~5 成功率之间存在一致性约束。报告汇总时使用该关系检查结果是否合理。

## 3. 结果来源与整理方式

本报告汇总当前目录中可解析的 CALVIN ABC→D top-level results：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/*/reports/**/results.json
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/**/calvin_eval/merged_results.json
```

整理规则如下：

- worker 级 `worker_*/results.json` 只作为聚合输入，不作为独立模型结果列入主表。
- 主表按 `Average Chain Length` 从高到低排序。
- `n=10`、`n=50`、`n=100` 结果用于观察快速趋势；`n=300`、`n=1000` 更适合做稳定比较。
- `Near Miss` 表示接近成功但未通过环境判定的比例，用于判断模型是完全失败还是动作精度、接触时序或终止条件不足。

## 4. CALVIN ABC→D 完整结果

| Model / Run | n | Avg Len | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Near Miss | Related Near Miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WMH 8k state8 connector smoke | 10 | 2.000 | 70.0% | 60.0% | 40.0% | 20.0% | 10.0% | 11.1% | 0.0% |
| WMH MoE95k LoRA Aug latest | 100 | 1.940 | 77.0% | 55.0% | 35.0% | 18.0% | 9.0% | 20.9% | 3.3% |
| GTY MoE 60k augmented | 100 | 1.910 | 76.0% | 54.0% | 32.0% | 17.0% | 12.0% | - | - |
| WMH MoE95k LoRA Aug latest | 300 | 1.863 | 72.0% | 48.0% | 33.3% | 20.3% | 12.7% | 22.5% | 8.0% |
| WMH augmented hard v2 | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% | 11.4% | 5.3% |
| WMH MoE95k LoRA Mirror latest | 100 | 1.740 | 72.0% | 49.0% | 28.0% | 15.0% | 10.0% | 20.0% | 8.9% |
| WMH MoE95k LoRA Mirror latest | 300 | 1.670 | 72.3% | 44.7% | 28.3% | 14.0% | 7.7% | 23.5% | 5.1% |
| GTY MoE posttrain 95k | 300 | 1.640 | 72.7% | 42.0% | 24.0% | 15.7% | 9.7% | 21.4% | 8.9% |
| WMH LoRA2000 compare | 300 | 1.630 | 64.0% | 43.7% | 27.3% | 18.3% | 9.7% | 11.8% | 5.5% |
| GTY MoE quick | 50 | 1.540 | 78.0% | 40.0% | 24.0% | 8.0% | 4.0% | - | - |
| WMH adaptive MoE | 300 | 1.397 | 67.7% | 38.0% | 19.0% | 10.3% | 4.7% | 22.0% | 9.1% |
| WMH state8 connector steps8000 fast w4x8 | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% | 9.2% | 5.0% |
| WMH base8k | 300 | 1.050 | 54.0% | 25.7% | 13.7% | 8.0% | 3.7% | 10.0% | 4.8% |
| WMH 8k state8 connector | 100 | 0.990 | 50.0% | 27.0% | 12.0% | 6.0% | 4.0% | 8.3% | 5.2% |
| WMH ABC→D parallel | 1000 | 0.923 | 50.7% | 23.8% | 11.0% | 4.9% | 1.9% | - | - |
| WMH adaptive MoE 15k | 1000 | 0.910 | 50.9% | 23.4% | 10.4% | 4.4% | 1.9% | - | - |
| WMH debug GIF | 128 | 0.844 | 43.8% | 21.9% | 10.9% | 4.7% | 3.1% | 12.9% | 5.6% |
| WMH ABC→D parallel early | 1000 | 0.835 | 47.8% | 21.1% | 8.9% | 3.9% | 1.8% | - | - |
| LZH ensemble | 300 | 0.310 | 26.0% | 4.7% | 0.3% | 0.0% | 0.0% | 27.0% | 9.0% |
| local weighted checkpoint soup ensemble | 1000 | 0.302 | 25.8% | 3.8% | 0.6% | 0.0% | 0.0% | - | - |
| WMH ABC→D quick | 10 | 0.200 | 20.0% | 0.0% | 0.0% | 0.0% | 0.0% | - | - |
| local ordinary checkpoint eval | 1000 | 0.081 | 6.8% | 1.3% | 0.0% | 0.0% | 0.0% | - | - |
| WMH formal quick failed run | 10 | 0.000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 60.0% | 30.0% |
| WMH formal quick failed run early | 10 | 0.000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 30.0% | 10.0% |

## 5. 结果总览

### 5.1 样本规模视角

当前结果可以分成三类证据：

| 证据类型 | 代表结果 | 解释 |
|---|---|---|
| 中等规模效果证据 | WMH MoE95k LoRA Aug latest，`n=300`，Avg `1.863`，Task 5 `12.7%` | MoE + LoRA + augmentation 对长链中后段有效 |
| 大规模稳定性证据 | WMH state8 connector steps8000 fast w4x8，`n=1000`，Avg `1.086`，Task 5 `4.3%` | 样本数更大，更能刻画 ABC→D 的稳定泛化难度 |
| 小样本趋势证据 | GTY MoE 60k augmented，`n=100`，Avg `1.910`，Task 5 `12.0%` | 结果较强，但样本规模不同，不能与 `n=300/n=1000` 直接排序 |

### 5.2 MoE95k LoRA Aug 与 GTY MoE Posttrain 对比

同为 MoE/后训练相关路线，WMH MoE95k LoRA Aug latest 在 `n=300` 上优于 GTY MoE posttrain 95k：

| Model | n | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len |
|---|---:|---:|---:|---:|---:|---:|---:|
| GTY MoE posttrain 95k | 300 | 72.7% | 42.0% | 24.0% | 15.7% | 9.7% | 1.640 |
| WMH MoE95k LoRA Aug latest | 300 | 72.0% | 48.0% | 33.3% | 20.3% | 12.7% | 1.863 |

两者 Task 1 基本接近，但 WMH MoE95k LoRA Aug 在 Task 2~5 上更稳，说明该提升主要来自后续状态分布覆盖、任务切换稳定性和长链恢复能力，而不是单步视觉触发能力。

## 6. 指标形态与现象分析

### 6.1 长链衰减

以 WMH MoE95k LoRA Aug latest `n=300` 为例，survival curve 为：

$$
72.0\% \rightarrow 48.0\% \rightarrow 33.3\% \rightarrow 20.3\% \rightarrow 12.7\%.
$$

相邻条件成功率为：

$$
P(T_2\mid T_1)=\frac{48.0}{72.0}=66.7\%,
$$

$$
P(T_3\mid T_2)=\frac{33.3}{48.0}=69.4\%,
\quad
P(T_4\mid T_3)=\frac{20.3}{33.3}=61.0\%,
\quad
P(T_5\mid T_4)=\frac{12.7}{20.3}=62.3\%.
$$

这说明失败不是只发生在第一步。即使前一步成功，下一步仍有约 30% 到 40% 的条件失败率。长链评估暴露的是状态恢复、receding horizon 稳定性、任务切换和 gripper/contact 时序的联合瓶颈。

### 6.2 Augmentation 与后训练收益

`WMH base8k` 到 `WMH augmented hard v2` 的提升明显：

| Model | n | Avg Len | Task 5 |
|---|---:|---:|---:|
| WMH base8k | 300 | 1.050 | 3.7% |
| WMH augmented hard v2 | 300 | 1.847 | 12.0% |

Average Chain Length 提升 $0.797$，Task 5 提升 $8.3$ 个百分点。该结果说明 ABC→D 的主要瓶颈不是基础技能完全缺失，而是环境泛化、hard state 覆盖和长链中后段的分布偏移。

### 6.3 Mirror Augmentation 的边界

同样是 MoE95k LoRA 后训练，Aug 优于 Mirror：

| Variant | n | Avg Len | Task 5 |
|---|---:|---:|---:|
| Aug | 300 | 1.863 | 12.7% |
| Mirror | 300 | 1.670 | 7.7% |

Mirror 可以增强左右或空间对称性，但若 D 环境的失败更多来自物体状态、接触阶段和任务切换，而不只是左右视角变化，则 mirror augmentation 对长链帮助有限。Aug 的更高 Task 3/4/5 表明其覆盖了更多动作扰动、视觉扰动或 hard state。

### 6.4 Near Miss 的含义

MoE95k LoRA Aug `n=300` 的 near miss 为 `22.5%`，比 augmented hard v2 的 `11.4%` 高。这个现象有两层含义：

- 正向含义：模型已经学到目标区域和大致动作，只差末端精度或 gripper/contact 时序。
- 负向含义：模型可能在某些任务上反复接近但不完成，说明控制器、动作边界或 action head 精度仍不足。

因此 near miss 不是单纯的失败率补充项，而是判断后训练方向的重要信号。near miss 高时，precision-oriented post-training、separate gripper/contact head、action-bound layer、uncertainty/value-assisted selection 比继续单纯增加视觉增强更有针对性。

### 6.5 Ensemble 结果边界

local weighted checkpoint soup ensemble 在 `n=1000` 上 Avg Len 仅为 `0.302`，Task 5 为 `0.0%`。该结果说明，当前参与 ensemble 的模型不满足同构 checkpoint averaging 的基本前提：结构、初始化、训练阶段和 loss basin 都不完全一致。对 MoE/router/action head 直接做参数平均，会破坏专家分工和动作 mode。

这并不否定 action-level ensemble。更合理的形式是：

$$
\pi_{\mathrm{ens}}(a\mid x)
=\sum_m w_m(x)\pi_m(a\mid x),
$$

$$
w_m(x)
=\mathrm{softmax}\left(\frac{\mathrm{score}_m(x)}{\tau}\right).
$$

其中 $\mathrm{score}_m(x)$ 可以由 validation success、uncertainty、value head 或 task/domain router 给出。参数级 soup 与 action-level router 应作为不同实验范式分别评估。

## 7. 结果边界与实验缺口

本轮测评已经给出若干清晰趋势，但仍存在未闭合的实验边界：

- GTY MoE 60k augmented 只有 `n=100` 规模结果，缺少 `n=300/n=1000` 同规模验证。
- WMH MoE95k LoRA Aug latest 目前最强证据来自 `n=300`，缺少 `n=1000` 长评估。
- 参数级 ensemble 已显示退化，但同构 checkpoint averaging 与 action-level ensemble 尚未充分分离验证。
- Near miss 指标提示动作精度和 gripper/contact 时序仍是瓶颈，需要结合 failure replay、action-bound layer 和 contact-specific head 进一步分析。
- 世界模型路线已有短程 affordance 信号，但长链闭环控制能力尚未在 CALVIN ABC→D 上形成稳定优势。

## 8. 个人贡献

本报告中的个人贡献主要体现在工程实现、测评组织、结果整理、问题定位和分析归纳五个方面。

### 8.1 测评体系搭建与修复

- 贯通了交互式 CALVIN evaluation 流程，使 checkpoint 能通过 `interaction` 层启动 policy server、client rollout、日志记录和结果聚合。
- 修复和完善了 CALVIN evaluation 中与 `unnorm_key`、state/action 维度、checkpoint metadata、FAST token decode、policy server/client 并发和端口管理相关的问题，使评估失败可以更早定位。
- 将 worker 级结果与 top-level 聚合结果区分处理，避免把 `worker_*/results.json` 当成独立模型结果，减少统计口径混乱。
- 将 `n=10/50/100/300/1000` 等不同规模评估统一纳入同一结果表，并明确其统计含义差异。

### 8.2 指标定义与一致性检查

- 明确 CALVIN ABC→D 的 Task 1~5 是 survival success rate，而不是五个互相独立的分类指标。
- 写清楚 $\mathrm{AvgLen}=\sum_{k=1}^{5}\mathrm{SR}_k$ 的数学关系，用于解释平均链长与 Task 1~5 的一致性。
- 引入相邻条件成功率 $P(T_k\mid T_{k-1})$ 分析长链衰减，使结果解释从“哪个模型分数更高”转向“失败发生在链条哪个阶段”。
- 将 Near Miss 纳入结果分析，用于区分完全失败、接近成功但精度不足、以及重复接近但未完成三类现象。

### 8.3 结果汇总与模型对比

- 系统整理了公共成员目录和本地 run 目录中的 CALVIN ABC→D 结果，形成统一完整结果表。
- 对 MoE95k LoRA Aug、GTY MoE posttrain、adaptive MoE、state8 connector、base8k、world-model 相关结果和 ensemble 结果做了同表对比。
- 将不同样本规模结果拆分为中等规模效果证据、大规模稳定性证据和小样本趋势证据，避免不同 `n` 的结果被直接误排序。
- 分析了 augmentation、mirror augmentation、posttrain、MoE 和参数级 ensemble 对 Task 1~5 的不同影响。

### 8.4 Failure Pattern 与后训练方向分析

- 从测评结果中归纳出长链衰减、环境泛化、动作精度不足、gripper/contact 时序、任务切换退化和多峰动作平均等典型 failure pattern。
- 将 near miss 高的问题关联到 precision-oriented post-training、separate gripper/contact head、action-bound layer、uncertainty/value-assisted selection 等可执行改进方向。
- 将参数级 soup 的失败解释为异构 checkpoint、MoE/router 和 action head 不在同一 loss basin 下的结构性冲突，并进一步区分了同构 checkpoint averaging 与 action-level ensemble。
- 将 CALVIN failure sequences 与 `T20/T24/T27` 等 hard-task replay、advantage-weighted BC、DPO-style trajectory ranking 连接起来，为后训练闭环提供依据。

### 8.5 文档化与复现支持

- 将测评协议、指标定义、结果来源、完整结果表、指标形态分析、结果边界和个人贡献整合成独立报告。
- 使用 $...$ 和 $$...$$ 统一数学表达，使指标定义、条件成功率和 ensemble 公式更清晰。
- 将本报告与技术报告、工程使用文档、policy matrix 分离，避免评测结果、技术路线和操作指南混在同一个文件中。
- 保留完整结果表和路径来源，使后续复查、补充评估或复现实验有明确入口。
