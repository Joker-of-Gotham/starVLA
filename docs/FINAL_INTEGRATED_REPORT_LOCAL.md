# StarVLA SII SummerCamp 终极整合报告

> 本文件为最终汇总报告，整合工程实现、技术路线、测评结果、失败分析和个人贡献。分篇公开文档入口见 `docs/SUMMERCAMP_REPORT.md`。

## 摘要

本项目围绕 StarVLA 具身智能训练与评估体系进行了系统化工程实现与方法探索。核心贡献是将原本分散的训练脚本、模型选择、数据选择、动作头选择、后训练策略和测评流程统一为可组合的交互式 benchmark 系统。系统主流程被抽象为：

```text
dataset -> training policy -> base model -> action expert -> structure policy
```

该流程覆盖 CALVIN、OXE/Bridge、RoboTwin、RoboCasa、VLM-only robot image-language data 等数据入口，支持 Qwen3.5、Qwen3-VL、Cosmos-Predict2 等 base model，支持 OFT/MLP、FAST/action-token、PI/flow matching、GR00T、Cosmos world-model action expert、MoE/adaptive MoE 等动作专家，并通过 `T01-T28` 与 `S01-S32` 两组 policy matrix 支撑从头训练、续训、后训练、世界模型辅助训练、失败回放、offline RL、DPO-style ranking、distillation 和 ensemble。

在 CALVIN ABC→D 上，当前最强 `n=300` 结果为 **WMH MoE95k LoRA Aug latest**，Average Chain Length 为 `1.863`，Task 5 成功率为 `12.7%`；最完整 `n=1000` 正式规模结果为 **WMH state8 connector steps8000 fast w4x8**，Average Chain Length 为 `1.086`，Task 5 成功率为 `4.3%`。结果表明：增强与 MoE 后训练主要改善 Task 3~5 的长链保持能力，而不是单纯提高第一步 affordance detection。

## 1. 背景与目标

StarVLA 的目标是训练能够基于视觉、语言和机器人状态生成动作的通用机器人策略。形式化地，策略输入为当前观测 $o_t$、机器人状态 $s_t$、语言指令 $l$ 和历史信息 $h_t$，输出未来一段动作：

$$
\pi_\theta(o_t,s_t,l,h_t)\rightarrow a_{t:t+H-1}.
$$

CALVIN ABC→D 任务要求策略在 ABC 环境训练，并在未见 D 环境中连续完成 5 个 subtasks。它不只是单步操控任务，而是对视觉泛化、语言 grounding、动作精度、状态恢复、任务切换和长程误差控制的综合考验。

本项目的目标不是单次跑通某个 checkpoint，而是构建一个完整、可复现、可拓展的训练与评估系统，使不同数据集、模型、动作专家和训练策略可以真实组合运行，并能基于已训练 checkpoint 做后训练和优化。

## 2. 工程体系设计

### 2.1 目录与数据组织

工程约定数据、预训练模型、checkpoint 和 run log 分离存放：

```text
playground/Datasets/                 # 训练数据入口
playground/Pretrained_models/         # base model / tokenizer / processor
data/starvla/checkpoints/             # 训练输出 checkpoint
data/starvla/runs/                    # interaction run、日志、runner、评估结果
interaction/                          # 交互式训练、评估、监控、ensemble 入口
docs/                                 # 技术文档、测评报告、policy 矩阵
```

交互式训练通过 `interaction/bin/starvla-interact.sh` 进入，主菜单包含环境检查、catalog、性能检查、训练启动、评估启动、任务列表、监控、attach 和 stop 等功能。所有训练命令会生成 runner、日志、metadata 和 checkpoint 目录，便于复现。

### 2.2 五轴组合

系统将 VLA 训练拆成五个轴：

```text
dataset -> training policy -> base model -> action expert -> structure policy
```

这种拆分对应五类实际变量：

- dataset 决定 action/state 维度、horizon、camera、normalization、domain。
- training policy 决定 loss、采样、冻结策略、后训练、辅助目标。
- base model 决定视觉语言先验和算力成本。
- action expert 决定动作分布建模方式。
- structure policy 决定 connector、state encoder、MoE、world head、safety head、ensemble router。

系统支持从头训练，也支持从已有 checkpoint 做新 run 后训练。后训练不是简单 `continue`，而是：

$$
\theta_0=\mathrm{load}(\mathrm{checkpoint}),
$$

$$
\begin{aligned}
\min_\theta
&\quad \mathcal L_{\mathrm{new}}
{}+\lambda_{\mathrm{replay}}\mathcal L_{\mathrm{old}}\\
&\quad +\lambda_{\mathrm{KL}}\operatorname{KL}(\pi_\theta\Vert\pi_0)
{}+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{pref}}.
\end{aligned}
$$

这使 evaluation failure、偏好排序、success signal、hard state 和旧任务 replay 都能进入训练闭环。

## 3. 数学建模与训练目标

### 3.1 Embodiment-aware Normalization

异构机器人不能直接共享原始 action space。对 embodiment $e$ 的动作组 $g$，采用：

$$
\tilde a_g=\frac{a_g-\mu_{e,g}}{\sigma_{e,g}}
\quad \text{or} \quad
\tilde a_g=2\frac{a_g-\min_{e,g}}{\max_{e,g}-\min_{e,g}}-1.
$$

训练在 normalized local coordinates 中进行，部署时执行反归一化：

$$
a_g=\sigma_{e,g}\tilde a_g+\mu_{e,g}.
$$

这样可以避免高量纲 hand、arm、waist 或 gripper binary 因尺度不同而破坏共享梯度。

### 3.2 Continuous BC / OFT

OFT/MLP continuous BC 是第一基线，适合快速稳定训练：

$$
\begin{aligned}
\mathcal L_{\mathrm{OFT}}
&= \sum_t\sum_g w_g
\left\lVert \tilde a_{t,g}-f_\theta(c_t)_g\right\rVert_1
{}+\lambda_{\mathrm{grip}}\operatorname{BCE}(g_t,\hat g_t).
\end{aligned}
$$

该路线优势是推理快、诊断清楚、适合作为所有数据集的第一强 baseline；缺点是单峰 L1 容易产生 mode averaging。

### 3.3 FAST / Action-token

FAST 将 action chunk 转换到频域，再量化成 token：

$$
z=Q(\mathrm{DCT}(a_{t:t+H-1})).
$$

训练目标为：

$$
\begin{aligned}
\mathcal L_{\mathrm{FAST}}
&= -\sum_i\log p_\theta(z_i\mid c,z_{<i})\\
&\quad +\lambda_{\mathrm{rec}}\left\lVert a-\operatorname{Dec}(z)\right\rVert_1.
\end{aligned}
$$

其优势是降低 long-horizon token 长度；风险是 action-token embedding、tokenizer 和 detokenization 必须对齐，否则评估阶段会出现 token decode `None` 或动作失真。

### 3.4 Flow Matching / GR00T / PI

Flow matching 适合多峰动作分布。令 $x_0\sim\mathcal N(0,I)$、$x_1=\tilde a$、$\tau\sim U(0,1)$：

$$
x_\tau=(1-\tau)x_0+\tau x_1,
\qquad
u=x_1-x_0.
$$

目标为：

$$
\begin{aligned}
\mathcal L_{\mathrm{FM}}
&= \mathbb E\left[
\left\lVert v_\theta(x_\tau,\tau,c)-u\right\rVert_2^2
\right].
\end{aligned}
$$

相比 OFT，它能表达多种合法动作模式；代价是训练、采样和条件路径更复杂。

### 3.5 MoE Action Expert

MoE 用 router 将任务、动作组和 embodiment 映射到不同专家：

$$
y=\sum_{i\in\mathrm{TopK}}r_i(x,e,g)E_i(x),
\qquad
r=\mathrm{softmax}(R(x,e,g)).
$$

为避免专家塌缩，引入负载均衡项：

$$
\begin{aligned}
\mathcal L=\mathcal L_{\mathrm{action}}
&\quad +\lambda_{\mathrm{lb}}K\sum_i f_i p_i.
\end{aligned}
$$

CALVIN 中不同 subtask 对 contact、gripper、空间移动和旋转的要求明显不同，MoE 能减少单 head 的 mode interference，因此当前最强结果主要来自 MoE/LoRA/Aug 路线。

## 4. Training Policy 与 Structure Policy

本项目将训练策略和结构策略分开定义。

Training Policy `T01-T28` 覆盖：

- 数据 schema、normalization、mixture sampling。
- VLM-only pre/post-training。
- action-token embedding warmup 与 FAST tokenizer。
- OFT、flow matching、diffusion、hybrid residual。
- grouped action、gripper/contact、multi-objective cotrain。
- cross-embodiment cotrain、future representation alignment、Fast-WAM。
- DIAL latent intent、human/synthetic video retargeting。
- uncertainty、hard-task replay、anti-forgetting、augmentation、sim-to-real。
- offline RL、online RL、flow-based RL、DPO-style ranking、distillation。

Structure Policy `S01-S32` 覆盖：

- base VLM preservation、LoRA/adapter。
- action-token embedding、multi-camera connector。
- embodiment/FPS/control-mode tokens。
- proprio encoder、temporal memory。
- OFT/FAST/hybrid/FM/DiT action heads。
- bimanual、humanoid grouped decoder、gripper/contact head。
- hierarchical MoE、adapter-MoE、fast-slow dual transformer。
- latent intent、world/future latent、representation alignment。
- object-slot、spatial/3D connector、Qwen dynamic resolution。
- value/success/reward、safety projection、uncertainty、sampler acceleration。
- policy ensemble/router、Cosmos interface、speech/omni extension。

组合逻辑是：Training Policy 定义优化目标和训练阶段，Structure Policy 定义模型内部怎么承载这些目标。两者必须共同选择，否则容易出现“文档上可选但实际不可训练”的问题。

## 5. 世界模型路线

世界模型路线采用 Cosmos-Predict2 作为 Video2World backbone 或 frozen teacher。其核心思想是将未来表示作为 action expert 的辅助约束：

$$
z_{t+k}=W(o_t,a_{t:t+k}),
\qquad
a_{t:t+H}=A(z_t,z_{t+1:t+K},s_t,l).
$$

训练时可加入 future representation alignment：

$$
\begin{aligned}
\mathcal L_{\mathrm{future}}
&= \sum_k
\left\lVert g_\theta(c_t,a_{t:t+k})
{}-\mathrm{sg}(\phi(o_{t+k}))\right\rVert_2^2,
\end{aligned}
$$

$$
\begin{aligned}
\mathcal L=\mathcal L_{\mathrm{action}}
&\quad +\lambda_w\mathcal L_{\mathrm{future}}.
\end{aligned}
$$

该路线理论上有助于长程规划和环境动态建模，但 CALVIN ABC→D 对末端动作精度、gripper/contact 时序和闭环恢复要求很高。当前 Cosmos-Predict2-PI 训练中 `steps_5500` 是有效 top-k checkpoint，训练指标约为 `loss/action_dit_loss=0.231206`、`mse_score=0.006678`、`global_batch_size=512`。公共 Cosmos-Predict2-GR00T 小规模结果曾达到 Task 1 40% 以上，但 Task 4/Task 5 基本为 0，说明它已学习部分短程 affordance，但尚未完成长链闭环能力转化。

## 6. 训练系统与 H200 优化

当前硬件目标是 8 卡 H200。交互式系统加入了：

- H200 throughput profile：支持大 batch、prefetch、persistent workers。
- NCCL/DeepSpeed profile：显式启用 NVLink P2P、NCCL async error handling、TF32/bf16。
- GPU preflight：检查显存、已有 compute process、GPU util、NVLink、NCCL benchmark 工具。
- disk guard resume：在公共目录磁盘空间低于阈值时停止训练，高于阈值后自动恢复。
- tmux run management：训练、评估和监控均有独立 session、runner、log。
- checkpoint safety：识别 128B 等坏 checkpoint，避免从不完整权重恢复。

训练中曾出现 OOM、磁盘不足、残留 GPU 进程、FAST token decode `None`、eval server/client 数不匹配等问题。本分支对这些问题做了系统化修复，使问题能在 preflight 或早期日志中定位，而不是训练结束后才发现。

## 7. CALVIN ABC→D 测评结果

CALVIN ABC→D 的指标是 survival curve。令第 $i$ 条 sequence 完成长度为 $l_i\in\{0,1,2,3,4,5\}$：

$$
\begin{aligned}
\mathrm{SR}_k
&= \frac{1}{N}\sum_{i=1}^{N}\mathbb 1[l_i\ge k],\\
\mathrm{AvgLen}
&= \sum_{k=1}^{5}\mathrm{SR}_k.
\end{aligned}
$$

当前关键结果如下：

| Model / Run | n | Avg Len | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WMH MoE95k LoRA Aug latest | 300 | 1.863 | 72.0% | 48.0% | 33.3% | 20.3% | 12.7% |
| WMH augmented hard v2 | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% |
| GTY MoE posttrain 95k | 300 | 1.640 | 72.7% | 42.0% | 24.0% | 15.7% | 9.7% |
| WMH adaptive MoE | 300 | 1.397 | 67.7% | 38.0% | 19.0% | 10.3% | 4.7% |
| WMH state8 connector steps8000 fast w4x8 | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% |
| LZH ensemble | 300 | 0.310 | 26.0% | 4.7% | 0.3% | 0.0% | 0.0% |
| local weighted checkpoint soup ensemble | 1000 | 0.302 | 25.8% | 3.8% | 0.6% | 0.0% | 0.0% |

最强 `n=300` 结果是 WMH MoE95k LoRA Aug latest。它相对 GTY MoE posttrain 95k 的提升主要集中在 Task 3~5，说明增强和 LoRA 后训练改善的是长链中后段稳定性，而不是简单提升 Task 1 的识别能力。

## 8. 失败模式与根因分析

### 8.1 长链衰减

若 $q_k=P(l_i\ge k\mid l_i\ge k-1)$，则：

$$
\mathrm{SR}_k=\prod_{j=1}^{k}q_j.
$$

即使每一步条件成功率 $q_j\approx0.7$，五步全成功也只有：

$$
\mathrm{SR}_5\approx0.7^5=16.8\%.
$$

因此 Task 1 达到 70% 并不意味着 Task 5 会高。提升最终成绩必须提高每个条件转移，而不是只优化第一步。

### 8.2 Failure Pattern 分类

| 类别 | 典型表现 | 根因 | 改进方向 |
|---|---|---|---|
| F1 接口/归一化失败 | state/action 维度错、unnorm_key 不匹配 | metadata 与评估接口不一致 | metadata 映射、维度适配、preflight |
| F2 单步感知失败 | Task 1 低 | D 环境视觉 domain shift | VLM robot-domain pretrain、增强、object-slot |
| F3 动作精度不足 | near miss 高 | gripper/contact phase 或 action bound 不稳 | separate gripper、safety projection、failure replay |
| F4 长程误差累积 | Task 3~5 快速下降 | chunk drift、状态偏移、任务切换失败 | temporal memory、latent intent、value/success head |
| F5 环境泛化失败 | ABC 到 D 掉分 | 背景、初态、光照变化 | domain randomization、hard augmentation、adapter/MoE |
| F6 多峰动作平均 | 轨迹平滑但无效 | 单 head 或异构参数平均破坏 mode | MoE、flow/diffusion、action-level ensemble |
| F7 系统失败 | OOM、坏 checkpoint、残留进程 | 工程稳定性不足 | H200 profile、disk guard、checkpoint validation |

## 9. Ensemble 结论

本次尝试了参数级 checkpoint soup ensemble，但效果明显下降：local weighted checkpoint soup ensemble 在 `n=1000` 上 Avg Len 仅为 `0.302`，Task 5 为 `0.0%`。这说明参与 ensemble 的模型不是同构结构、同一初始化、同一 loss basin 的相邻 checkpoint。直接平均 MoE/router/action head 会破坏专家分工。

更合理的 ensemble 应在 action level 或 router level 做：

$$
\begin{aligned}
\pi_{\mathrm{ens}}(a\mid x)
&= \sum_m w_m(x)\pi_m(a\mid x),
\end{aligned}
$$

$$
\begin{aligned}
w_m(x)
&= \operatorname{softmax}\left(\frac{\mathrm{score}_m(x)}{\tau}\right).
\end{aligned}
$$

$\mathrm{score}_m(x)$ 可以来自 validation success、uncertainty、value head、task/domain router 或近期 rollout 成功率。由当前结果可见，需要把同构 checkpoint averaging、异构参数平均和 action-level online routing 分开讨论；三者的结构假设不同，不能用一个 ensemble 结论互相替代。

## 10. 个人贡献

本部分从工程实现、训练系统、测评闭环、问题修复、实验分析和文档整理六个角度总结个人贡献。

### 10.1 工程实现与系统贯通

本项目中，我将 StarVLA 原有分散的训练脚本、模型入口、数据入口、评估脚本和 checkpoint 管理方式，整理为统一的交互式系统。核心流程被抽象为：

```text
dataset -> training policy -> base model -> action expert -> structure policy
```

围绕该流程，我完成了 dataset、base model、action expert、training policy 和 structure policy 的 catalog 化，使用户可以在交互式界面中选择 CALVIN、OXE/Bridge、RoboTwin、RoboCasa、VLM-only robot data 等数据集，组合 Qwen3.5、Qwen3-VL、Cosmos-Predict2 等 base model，并自由切换 OFT、FAST、PI、GR00T、Cosmos action expert、MoE/adaptive MoE 等动作专家。

我还将 `T01-T28` 与 `S01-S32` 组织为完整 policy matrix，使训练策略和结构策略不再只是概念说明，而是能进入训练、后训练、评估和部署 metadata 的实际配置项。该工作使从头训练、继续训练、从已有 checkpoint 后训练、ensemble 初始化和评估之间形成统一闭环。

### 10.2 训练与后训练能力建设

在训练侧，我补强了从 base pretrained 开始训练和从已有 checkpoint 后训练的两条路径。后训练被明确定义为从 checkpoint 初始化新 run，并改变数据分布或目标函数，而不是简单恢复同一 optimizer 状态：

$$
\theta_0=\mathrm{load}(\mathrm{checkpoint}),
$$

$$
\begin{aligned}
\min_\theta
&\quad \mathcal L_{\mathrm{new}}
{}+\lambda_{\mathrm{replay}}\mathcal L_{\mathrm{old}}\\
&\quad +\lambda_{\mathrm{KL}}\operatorname{KL}(\pi_\theta\Vert\pi_0)
{}+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{pref}}.
\end{aligned}
$$

围绕该目标，我在交互式流程中补充了 hard-task replay、anti-forgetting、advantage-weighted BC、DPO-style trajectory ranking、EMA/SWA/checkpoint averaging、world/future latent auxiliary 等后训练入口，使已经训练出的模型可以继续基于失败样本、偏好信号、旧任务 replay 和辅助 representation loss 做系统优化。

### 10.3 H200 多卡训练与稳定性优化

针对当前八卡 H200 配置，我完善了训练性能检查和启动流程，包括 GPU 显存检查、已有 compute process 检查、NVLink/NCCL 可用性检查、DeepSpeed/Accelerate 配置选择、bf16/TF32、dataloader worker、prefetch、persistent worker 和大 batch 配置。

我处理了训练过程中的 OOM、坏 checkpoint、磁盘空间不足、残留 GPU 进程、NCCL 退出和 tmux runner 管理问题，并实现了磁盘守护恢复脚本：当公共目录空间不足时停止训练，空间恢复后自动从最近完整 checkpoint 继续。该机制降低了长时间训练因共享磁盘波动中断的风险。

### 10.4 CALVIN 测评修复与结果整理

在测评侧，我贯通了 CALVIN websocket evaluation 的 server/client 流程，修复了 normalization key、state/action dimension、policy wrapper、FAST token decode、server/client 并发、总 worker 数和结果聚合等问题，使 checkpoint 可以稳定进入 CALVIN ABC→D 评估。

我整理了公共成员目录和本地 run 目录中的 top-level result，并区分 worker 级结果和聚合结果，形成统一的 CALVIN ABC→D 结果表。测评指标被整理为 survival success rate：

$$
\begin{aligned}
\mathrm{SR}_k
&= \frac{1}{N}\sum_{i=1}^{N}\mathbb 1[l_i\ge k],\\
\mathrm{AvgLen}
&= \sum_{k=1}^{5}\mathrm{SR}_k.
\end{aligned}
$$

基于该定义，我进一步分析了 Task 1~5 的长链衰减、相邻条件成功率、Near Miss、augmentation 收益、MoE 后训练收益和 ensemble 退化原因。

### 10.5 世界模型、小模型和 Ensemble 路线分析

我尝试并分析了 Cosmos-Predict2 世界模型路线，将其定位为 Video2World backbone、frozen teacher 或 future latent regularizer。该路线的理论目标是引入未来状态表示：

$$
\begin{aligned}
\mathcal L
&= \mathcal L_{\mathrm{action}}
{}+\lambda_w\mathcal L_{\mathrm{future}},
\end{aligned}
$$

其中：

$$
\begin{aligned}
\mathcal L_{\mathrm{future}}
&= \sum_k
\left\lVert g_\theta(c_t,a_{t:t+k})
{}-\mathrm{sg}(\phi(o_{t+k}))\right\rVert_2^2.
\end{aligned}
$$

我同时对小模型、小 step、小 batch 和快速 smoke 结果进行了整理，区分 pipeline 验证结果、趋势结果和正式规模结果。对于 ensemble，我实现并评估了参数级 checkpoint soup，结果显示异构 MoE/router/action head 直接平均会显著退化。由此我将 ensemble 方案进一步区分为同构 checkpoint averaging、异构参数平均和 action-level router 三种不同范式。

### 10.6 文档、报告与复现材料

我完成了 README、工程使用文档、policy matrix、技术报告、CALVIN 测评报告和本终极整合报告的整理。文档中明确写清楚了环境离线配置、数据放置、模型放置、checkpoint 组织、交互式训练、续训、后训练、评估、ensemble、磁盘守护和 H200 多卡训练方式。

在报告表达上，我将核心数学公式统一为 `$...$` 和 `$$...$$`，使 normalization、BC loss、FAST tokenization、flow matching、world latent loss、survival curve、DPO-style ranking 和 ensemble router 的数学逻辑更清楚。最终形成了“工程系统 + 理论方法 + 测评结果 + 失败分析 + 个人贡献”的完整材料链路。

## 11. 总结

本项目完成了三个层面的建设：

1. **工程系统贯通。** 数据、模型、训练策略、结构策略、后训练、评估和 ensemble 被统一到 interaction 流程中，支持可复现 run、日志、checkpoint、monitor 和恢复。
2. **方法体系扩展。** 通过 `T01-T28` 和 `S01-S32` 覆盖 normalization、sampling、OFT、FAST、flow/diffusion、MoE、world model、offline RL、DPO、distillation、safety、uncertainty、ensemble 等关键路线。
3. **结果分析闭环。** 基于 CALVIN ABC→D Task 1~5 和 Average Chain Length，对 MoE、augmentation、world model、小模型、ensemble 的有效性和失败模式进行了量化分析。

当前仍未闭合的技术问题包括：不同样本规模下 MoE/Aug 结果的统计方差；failure replay、advantage-weighted BC 和 DPO-style ranking 对 Task 3~5 的实际增益；near miss 高的模型能否通过 gripper/contact 精度后训练转化为真实 success；world model 作为 future latent regularizer 与直接 action backbone 的边界。
