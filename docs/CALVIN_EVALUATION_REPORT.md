# CALVIN ABC→D 测评报告

本文只记录 CALVIN ABC→D 测评协议、完整结果表和基于结果的分析。技术路线、失败模式根因和改进方案见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。

## 1. 测评协议

CALVIN ABC→D 的核心检验是：策略只在 ABC 环境或 ABC 相关数据上训练，然后在未见过的 D 环境中完成由 5 个子任务组成的长程指令链。每条 evaluation sequence 最多包含 5 个 subtasks，因此报告包含：

- `Task 1` 成功率：至少完成第 1 个 subtask 的 sequence 比例。
- `Task 2` 成功率：连续完成前 2 个 subtasks 的 sequence 比例。
- `Task 3` 成功率：连续完成前 3 个 subtasks 的 sequence 比例。
- `Task 4` 成功率：连续完成前 4 个 subtasks 的 sequence 比例。
- `Task 5` 成功率：连续完成全部 5 个 subtasks 的 sequence 比例。
- `Average Chain Length`：每条 sequence 平均完成的 subtask 数。

令第 $i$ 条 sequence 完成长度为 $l_i\in\{0,1,2,3,4,5\}$，总数为 $N$。第 $k$ 个位置的 survival success rate 为：

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

因此 `Average Chain Length` 不只是单独指标，它也等于 Task 1 到 Task 5 五个 survival rate 的和。这个性质用于检查结果是否一致。

## 2. 结果来源

本报告汇总了当前目录中可解析的 CALVIN ABC→D top-level results：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/*/reports/**/results.json
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/**/calvin_eval/merged_results.json
```

worker 级 `worker_*/results.json` 只作为聚合输入，不作为独立模型结果列入主表。下表按 `Average Chain Length` 从高到低排序。`n=10` 或 `n=50/100` 的结果只作为快速趋势或 smoke test；正式比较以 `n=300` 或 `n=1000` 为主要依据。

## 3. CALVIN ABC→D 完整结果表

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

## 4. 主要结论

当前最强的 `n=300` 结果是 **WMH MoE95k LoRA Aug latest**，平均链长 `1.863`，Task 5 为 `12.7%`。它相比 GTY MoE posttrain 95k 的 `1.640 / 9.7%` 更强，主要提升在 Task 3、Task 4、Task 5：

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len |
|---|---:|---:|---:|---:|---:|---:|
| GTY MoE posttrain 95k | 72.7% | 42.0% | 24.0% | 15.7% | 9.7% | 1.640 |
| WMH MoE95k LoRA Aug latest | 72.0% | 48.0% | 33.3% | 20.3% | 12.7% | 1.863 |

这说明增强和 LoRA 后训练没有显著提高第一步感知触发能力，但明显改善了长链中段和末段。换言之，收益主要来自更稳的连续控制和更好的跨任务状态分布覆盖，而不是单步 affordance 的简单提升。

当前最完整的 `n=1000` 正式规模结果是 **WMH state8 connector steps8000 fast w4x8**，平均链长 `1.086`，Task 5 为 `4.3%`。虽然它不如 `n=300` 的最强 MoE95k 结果，但样本数更大，方差更低，能更真实地反映 ABC→D 的泛化难度。

`n=100` 的 GTY MoE 60k 结果达到 `1.910`、Task 5 `12.0%`，但样本数较小。由于 CALVIN sequence 分布中任务组合差异较大，`n=100` 结果可能有 2 到 5 个百分点的波动，因此它更适合作为趋势信号，而不是与 `n=300/n=1000` 结果直接排序。

参数级 ensemble 没有带来提升。local weighted checkpoint soup ensemble 在 `n=1000` 上只有 `0.302` 平均链长，Task 5 为 `0.0%`。这说明当前候选模型不是同一 loss basin 的同构 checkpoint，直接平均 MoE/router/action head 权重会破坏专家分工和动作 mode。

## 5. 指标形态分析

### 5.1 长链衰减

CALVIN ABC→D 的核心困难不是只完成 Task 1，而是 survival curve 是否缓慢衰减。以 WMH MoE95k LoRA Aug latest `n=300` 为例：

$$
72.0\% \rightarrow 48.0\% \rightarrow 33.3\% \rightarrow 20.3\% \rightarrow 12.7\%.
$$

相邻条件成功率可粗略估计为：

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

这条曲线说明失败不是只发生在第一步；即使前一步成功，下一步仍有 30% 到 40% 的条件失败率。改进重点应放在跨 subtask 的状态恢复、receding horizon 稳定性、任务切换和 gripper/contact 时序。

### 5.2 增强与后训练收益

`WMH base8k` 到 `WMH augmented hard v2` 的提升很明显：

| Model | n | Avg Len | Task 5 |
|---|---:|---:|---:|
| WMH base8k | 300 | 1.050 | 3.7% |
| WMH augmented hard v2 | 300 | 1.847 | 12.0% |

Average Chain Length 提升 $0.797$，Task 5 提升 $8.3$ 个百分点。这个结果支持两个判断：

- ABC→D 的主要瓶颈是环境泛化和状态分布偏移，不是模型完全没有基本技能。
- hard augmentation 或 failure-like 数据重采样对长链后半段更有效，因为后半段状态更偏离 demonstrations 的初始分布。

### 5.3 MoE95k LoRA Aug 与 Mirror 的差异

同样是 MoE95k LoRA 后训练，Aug 优于 Mirror：

| Variant | n | Avg Len | Task 5 |
|---|---:|---:|---:|
| Aug | 300 | 1.863 | 12.7% |
| Mirror | 300 | 1.670 | 7.7% |

Mirror 可以增加左右/空间对称性，但如果 CALVIN D 环境的失败更多来自物体状态、接触阶段和任务切换，而不只是左右视角变化，则 mirror augmentation 对长链帮助有限。Aug 的高 Task 3/4/5 表明它更好覆盖了动作扰动、视觉扰动或 hard state。

### 5.4 Near Miss 解释

Near Miss 高说明模型经常接近成功状态但未通过环境判定。MoE95k LoRA Aug `n=300` 的 near miss 为 `22.5%`，比 augmented hard v2 的 `11.4%` 高。这可以有两种解释：

- 正向解释：模型已学到目标区域和大致动作，只差末端精度或 gripper/contact 时序。
- 负向解释：模型在某些任务上产生重复接近但不完成的轨迹，说明控制器或 action head 精度不足。

因此 near miss 高的模型适合做 precision-oriented post-training，例如 `T20` failure replay、`T24` advantage-weighted BC、`S27` safety/action-bound、`S28` uncertainty/value-assisted selection，而不是只增加视觉增强。

## 6. 结果边界与实验缺口

当前结果可以分成三类证据：

- **中等规模效果证据**：WMH MoE95k LoRA Aug latest 在 `n=300` 上达到 Avg `1.863`、Task 5 `12.7%`，说明 MoE + LoRA + augmentation 对长链中后段有效。
- **大规模稳定性证据**：WMH state8 connector steps8000 fast w4x8 在 `n=1000` 上达到 Avg `1.086`、Task 5 `4.3%`，样本数更大，能更稳定地刻画 ABC→D 泛化难度。
- **小样本趋势证据**：GTY MoE 60k augmented 在 `n=100` 上达到 Avg `1.910`、Task 5 `12.0%`，说明该路线值得纳入对比，但该结果尚不具备与 `n=300/n=1000` 结果同等的统计稳定性。

尚未闭合的实验缺口包括：

- GTY MoE 60k augmented 缺少 `n=300/n=1000` 同规模验证。
- WMH MoE95k LoRA Aug latest 缺少 `n=1000` 长评估。
- 当前参数级 ensemble 结果显示异构 soup 会破坏动作专家结构，ensemble 结论需要与同构 checkpoint averaging 或 action-level ensemble 区分讨论。
