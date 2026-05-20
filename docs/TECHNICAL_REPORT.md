# StarVLA SII SummerCamp 技术报告

本文聚焦 StarVLA 在本分支中的技术路线、模型结构、训练策略、后训练逻辑、失败模式和改进方案。CALVIN ABC→D 的完整 Task 1~5 测评结果见 [CALVIN_EVALUATION_REPORT.md](CALVIN_EVALUATION_REPORT.md)，工程操作流程见 [SUMMERCAMP_STARVLA.md](SUMMERCAMP_STARVLA.md)，完整 policy 矩阵见 [POLICY_MATRIX.md](POLICY_MATRIX.md)。

## 1. 问题定义

CALVIN ABC→D 的核心目标是：在训练环境 ABC 上学习机器人策略，并在未见过的 D 环境中完成 5-step long-horizon instruction chain。输入包含图像观测 $o_t$、语言指令 $l$、机器人状态 $s_t$ 和必要的历史信息，输出是未来一段动作 chunk：

$$
\pi_\theta(o_t, s_t, l, h_t) \rightarrow a_{t:t+H-1}.
$$

StarVLA 的参数可以拆成：

$$
\theta=\{\theta_B,\theta_C,\theta_E,\theta_A,\theta_{\mathrm{aux}}\}.
$$

其中 $\theta_B$ 是 base VLM 或 world backbone，$\theta_C$ 是视觉/语言/action connector，$\theta_E$ 是 state/proprio/embodiment encoder，$\theta_A$ 是 action expert，$\theta_{\mathrm{aux}}$ 是 value、success、world、uncertainty 等辅助头。

训练目标不能直接在异构原始 action space 上做统一回归，而应先进入 embodiment-aware normalized action space。对第 $e$ 个 embodiment 的第 $g$ 个动作组，归一化可以写成：

$$
\tilde a_g=\frac{a_g-\mu_{e,g}}{\sigma_{e,g}}
\quad \text{or} \quad
\tilde a_g = 2\frac{a_g-\min_{e,g}}{\max_{e,g}-\min_{e,g}}-1.
$$

连续动作行为克隆的主损失为：

$$
\mathcal L_{\mathrm{BC}}
= \sum_g w_g \left\lVert \pi_\theta(x)_g-\tilde a_g \right\rVert_1
+ \lambda_g \,\mathrm{BCE}(\hat g,g).
$$

部署时再反归一化：

$$
a_g=\sigma_{e,g}\tilde a_g+\mu_{e,g}.
$$

这个设计是多机器人、多动作空间训练的前提。若不做 group-wise normalization，高量纲 hand、arm、waist 或 gripper binary 会以不同尺度进入同一个 loss，导致共享 trunk 的梯度方向被大尺度 action group 主导。

## 2. 技术路线选择

### 2.1 可组合 VLA 系统

本次工程目标不是只跑通一个 checkpoint，而是构建可复现 benchmark pipeline。因此系统被拆成五个可组合轴：

```text
dataset -> training policy -> base model -> action expert -> structure policy
```

这个拆分对应 VLA 训练中的五类不确定性：

- dataset 决定 action/state/horizon、camera、normalization 和 domain shift。
- training policy 决定采样、loss、冻结策略、后训练和 evaluation-driven optimization。
- base model 决定视觉语言先验和计算成本。
- action expert 决定动作分布建模方式。
- structure policy 决定 connector、state encoder、MoE、world branch、safety/uncertainty 等结构能力。

这种设计使 `qwen3vl_4b_action + QwenOFT`、`qwen3vl_4b_action + QwenFast`、`qwen3vl_4b_action + QwenPI`、`cosmos_predict2 + CosmoPredict2PI` 等组合都能走同一训练、续训、后训练和评估入口。

### 2.2 Base Model 路线

选择 Qwen3.5/Qwen3-VL 作为主线，是因为 CALVIN ABC→D 的瓶颈同时包含视觉定位、语言条件和连续控制。Qwen3-VL-4B 提供较强视觉语言表示，`Action` 版本提供 action-token embedding 扩展，适合 FAST、OFT、PI 三类 action expert。小模型 Qwen3.5-0.8B/2B 用于快速验证 pipeline，大模型 Qwen3.5-4B/9B 或 Qwen3-VL-4B 用于最终效果。

Cosmos-Predict2 世界模型路线也被纳入，因为长程任务理论上受益于未来 latent 和环境动力学建模。它的条件分解更接近：

$$
p(a_{t:t+H}\mid o_t,l,s_t)
= \int p_A(a_{t:t+H}\mid z_{\mathrm{future}},s_t,l)
\,p_W(z_{\mathrm{future}}\mid o_t,l)\,dz_{\mathrm{future}}.
$$

但在本次时间预算下，世界模型骨干训练和推理成本较高，action expert 还需要更长后训练才能把 future latent 转成稳定控制。因此本次主要有效结果集中在 Qwen/MoE VLA 路线，世界模型保留为探索性分支。

### 2.3 Action Expert 路线

OFT/MLP continuous BC 是第一 baseline，因为它直接优化连续 action chunk，训练稳定、推理快、易评估：

$$
\mathcal L_{\mathrm{OFT}}
= \sum_t \sum_g w_g
\left\lVert \tilde a_{t,g}-f_\theta(c_t)_g \right\rVert_1
+ \lambda_{\mathrm{grip}}\mathrm{BCE}(g_t,\hat g_t).
$$

FAST/action-token 路线适合把长 horizon 动作压缩成更短 token 序列：

$$
z=Q(\mathrm{DCT}(a_{t:t+H-1})),
$$

$$
\mathcal L_{\mathrm{FAST}}
=-\sum_i \log p_\theta(z_i\mid c,z_{<i})
+\lambda_{\mathrm{rec}}\left\lVert a-\mathrm{Dec}(z)\right\rVert_1.
$$

它理论上能降低自回归长度，但对 action-token embedding 和 tokenizer 对齐敏感；如果 token decode 输出 `None`，说明模型没有形成有效 action-token distribution，评估会直接失败。

PI/GR00T flow matching 适合多模态动作分布。令 $x_0\sim\mathcal N(0,I)$、$x_1=\tilde a$、$\tau\sim U(0,1)$，构造：

$$
x_\tau=(1-\tau)x_0+\tau x_1,
\qquad
u=x_1-x_0.
$$

flow matching 损失为：

$$
\mathcal L_{\mathrm{FM}}
=\mathbb E\left[
\left\lVert v_\theta(x_\tau,\tau,c)-u \right\rVert_2^2
\right].
$$

相比单点 L1 regression，flow/diffusion 可以表达多峰动作；代价是训练、采样和 condition path 稳定性要求更高。

MoE action expert 是本次结果中最有效的结构方向之一。其核心是用 router 把不同任务、动作 group 或 embodiment 分配给不同 expert：

$$
y=\sum_{i\in\mathrm{TopK}} r_i(x,e,g)E_i(x),
\qquad
r=\mathrm{softmax}(R(x,e,g)).
$$

为避免少数 expert 被过度使用，引入 load-balance 正则：

$$
\mathcal L=\mathcal L_{\mathrm{action}}
+\lambda_{\mathrm{lb}}K\sum_i f_i p_i.
$$

CALVIN 中 open drawer、move slider、rotate block、lift block、stack block 对动作精度和 contact phase 的要求不同，单一 MLP/head 容易把 mode 平均掉。MoE 通过专家分工降低 mode interference，因此在 Task 3~5 上更有优势。

## 3. 训练策略设计

### 3.1 数据与采样

多数据源训练不能直接按样本数拼接。若 $n_i$ 是第 $i$ 个数据源大小，采样分布使用温度重加权：

$$
p(D_i)\propto n_i^\alpha,\qquad 0<\alpha<1.
$$

$\alpha$ 越小，小数据集权重越高。训练目标从普通 empirical risk：

$$
\min_\theta \mathbb E_{D_{\mathrm{all}}}\mathcal L(\theta)
$$

变成重加权 mixture risk：

$$
\min_\theta \sum_i p(D_i)\,
\mathbb E_{(x,a)\sim D_i}\mathcal L_i(\theta).
$$

这对 OXE、Bridge、Fractal、RoboTwin、RoboCasa 这类规模差异很大的 mixture 必须显式处理，否则大数据源会淹没小但关键的任务。

### 3.2 Action Chunking 与误差累积

若一步策略误差为 $\epsilon_t$，环境动力学局部 Lipschitz 常数为 $L_f$，则 open-loop 多步误差可粗略写成：

$$
\left\lVert \delta s_{t+k}\right\rVert
\le
\sum_{j=0}^{k-1} L_f^{k-1-j}
\left\lVert \epsilon_{t+j}\right\rVert .
$$

长程 CALVIN 的失败通常不是单步输出完全错误，而是若干小偏差在接触、抓取、释放阶段被放大。Action chunking 通过一次预测 $H$ 步动作减少频繁重新规划的噪声，但过长 chunk 又会增加 open-loop drift。因此采用 horizon curriculum：

```text
CALVIN: 4 -> 8
OXE: 8 -> 16
RoboTwin: 16 -> 32 -> 50
```

部署时用 receding horizon，只执行前若干步再重新预测，以平衡稳定性和闭环纠错。

### 3.3 后训练设计

后训练不是 `run-mode continue`。`continue` 续的是同一个 optimizer、scheduler 和 RNG 状态；后训练是从已有 checkpoint 初始化新 run，并改变数据分布或目标函数：

$$
\theta_0=\mathrm{load}(\mathrm{checkpoint}),
$$

$$
\min_\theta
\mathcal L_{\mathrm{new}}
+\lambda_{\mathrm{replay}}\mathcal L_{\mathrm{old}}
+\lambda_{\mathrm{KL}}\mathrm{KL}(\pi_\theta\Vert\pi_0)
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{pref}}.
$$

本分支重点支持：

- `T20` hard-task curriculum：按 failure rate 上采样失败任务。
- `T21` anti-forgetting：保留 replay、KL 和 hidden distillation，避免新数据适配破坏原技能。
- `T24` advantage-weighted BC：从 success/failure 估计 $A(s,a)$，用 $w=\exp(A/\eta)$ 加权 BC。
- `T27` DPO-style trajectory ranking：用成功轨迹和失败轨迹对构造偏好损失。
- `T28` checkpoint averaging / distillation：将 top-k 或 ensemble 转成单个部署模型。

这些策略的共同目标是把 evaluation signal 转回训练分布，而不是盲目增加 SFT steps。

## 4. 世界模型路线分析

世界模型路线用 Cosmos-Predict2 作为 Video2World backbone 或 frozen teacher。其理论优势在于把策略学习拆成环境动态建模和动作反解：

$$
z_{t+k}=W(o_t,a_{t:t+k}),
\qquad
a_{t:t+H}=A(z_t,z_{t+1:t+K},s_t,l).
$$

训练时可用 future representation alignment：

$$
\mathcal L_{\mathrm{future}}
=\sum_k
\left\lVert g_\theta(c_t,a_{t:t+k})
-\mathrm{sg}(\phi(o_{t+k}))\right\rVert_2^2,
$$

$$
\mathcal L=\mathcal L_{\mathrm{action}}
+\lambda_w\mathcal L_{\mathrm{future}}.
$$

这相当于用 future latent 作为 representation regularizer，迫使 action expert 的 hidden states 保留与未来状态相关的信息。优点是对长程规划和环境泛化有理论吸引力；缺点是计算成本高，且 CALVIN 对末端控制精度敏感，future latent 不能自动保证 gripper/contact 时序正确。

本地 Cosmos-Predict2-PI 训练中 `steps_5500` 是有效 top-k checkpoint，训练指标约为：

```text
loss/action_dit_loss: 0.231206
mse_score: 0.006678
global_batch_size: 512
epoch: 2.6356 / total_epochs: 3.5817
```

公共 Cosmos-Predict2-GR00T 小规模结果曾达到 Task 1 40% 以上，但 Task 4/Task 5 基本为 0。这说明世界模型分支学到了部分短程 affordance，但缺少足够 action expert 后训练来完成长链闭环。因此本次最终路线选择 Qwen/MoE VLA，世界模型保留为中后期增强方向。

## 5. Failure Pattern 分析

### 5.1 完整分类

| 类别 | 表现 | 根因 | 对应改进 |
|---|---|---|---|
| F1 接口/归一化失败 | state/action 维度错、`unnorm_key` 不匹配、token decode `None` | 评估接口和 checkpoint metadata 不一致 | 自动 metadata 映射、维度适配、FAST token preflight |
| F2 单步感知失败 | Task 1 低，模型找不到物体或目标区域 | D 环境视觉 domain shift、语言 grounding 弱 | `T03/T22` robot VLM pretrain 和增强，`S23/S24` object/spatial connector |
| F3 动作精度不足 | 接近目标但未触发 success，near miss 高 | L1 平均轨迹、gripper/contact phase 错、action bound 不稳 | `T11/T19/T20` gripper/contact/uncertainty/failure replay，`S15/S27/S28` |
| F4 长程误差累积 | Task 1/2 可行，Task 3~5 快速下降 | chunk drift、状态分布偏移、任务切换失败 | `T07/T20/T21/T24/T27`，`S06/S20/S26` |
| F5 环境泛化失败 | ABC 训练到 D 测试显著掉分 | 背景、物体初态、光照、遮挡分布变化 | `T22/T23` domain randomization，hard augmentation，adapter/MoE |
| F6 多模态动作平均 | 轨迹平滑但无效，MoE soup 退化 | 单 head 或参数平均把多个 action mode 混合 | MoE/router、flow/diffusion、action-level ensemble |
| F7 训练系统失败 | OOM、磁盘不足、残留进程、坏 checkpoint | 工程稳定性不足 | H200 profile、磁盘守护、top-k/full-state、GPU orphan cleanup |

### 5.2 长链失败的数理解释

CALVIN 指标本质上是 survival curve。若 $q_k=P(l_i\ge k\mid l_i\ge k-1)$，则：

$$
\mathrm{SR}_k=\prod_{j=1}^{k}q_j,
\qquad
\mathrm{AvgLen}=\sum_{k=1}^{5}\mathrm{SR}_k.
$$

即使每一步条件成功率看似不低，例如 $q_j\approx 0.7$，五步全成功也只有：

$$
\mathrm{SR}_5\approx 0.7^5=16.8\%.
$$

这解释了为什么 Task 1 70% 以上的模型，Task 5 仍可能只有 10% 左右。提升最终分数不能只优化第一步 success，而要提高每个条件转移 $q_k$，尤其是任务切换后的状态恢复能力。

### 5.3 典型失败案例

**误差累积。** 模型完成第一步后，末端位置偏离 demonstration 分布，下一步 VLM 仍能理解指令，但 action expert 的 state/action condition 已经落在训练分布外，导致后续动作持续偏移。对应改进是 action chunk curriculum、failure replay 和 value/success head。

**环境泛化失败。** ABC 到 D 后，视觉背景、物体初态或相机观测出现偏移。模型能在 ABC 中识别 drawer、slider、block，在 D 中 grounding 不稳。对应改进是 robot-domain VLM-only 预训练、颜色/crop/camera dropout、spatial/object-slot connector。

**长程规划退化。** 模型在每个 subtask 内能局部动作，但在 subtask 边界不能重置 intent，导致继续执行上一阶段动作。对应改进是 latent intent bottleneck、temporal memory、success/progress head 和 DPO-style trajectory ranking。

**动作精度不足。** lift、rotate、stack、place 对接触和 gripper 时序要求高，模型常出现 near miss。对应改进是 separate gripper/contact head、grouped action loss、uncertainty head、safety projection 和 hard-task post-training。

**多峰动作平均。** 同一语言目标可能有多种合法 approach direction。OFT/L1 或异构参数 soup 会输出 mode average，导致轨迹处在两个有效模式之间。对应改进是 MoE、flow/diffusion 或 action-level ensemble。

## 6. 结果驱动的技术判断

测评结果显示，短程能力和长程能力不是同一个瓶颈。`WMH base8k` 在 `n=300` 上 Avg `1.050`、Task 5 `3.7%`，而 `WMH augmented hard v2` 达到 Avg `1.847`、Task 5 `12.0%`。这说明基础技能已经存在，主要问题是 D 环境泛化、hard state 覆盖和 long-horizon recovery。

MoE95k LoRA Aug 的 `n=300` 结果为 Avg `1.863`、Task 5 `12.7%`，优于 GTY MoE95k 的 Avg `1.640`、Task 5 `9.7%`。其提升集中在 Task 3~5，说明后训练和增强对长链中后段有效，而不是仅提升第一步 detection。

异构 checkpoint soup 的 `n=1000` Avg 只有 `0.302`，Task 5 为 `0.0%`。这验证了参数级 ensemble 的适用边界：它要求同构结构、相近初始化和同一 basin。MoE/router/action head 不同的模型直接平均，会破坏专家路由和动作 mode。后续应改用 action-level ensemble：

$$
\pi_{\mathrm{ens}}(a\mid x)
=\sum_m w_m(x)\pi_m(a\mid x),
\qquad
w_m(x)=\mathrm{softmax}\left(\frac{\mathrm{score}_m(x)}{\tau}\right).
$$

其中 $\mathrm{score}_m$ 可来自 validation success、uncertainty、value head 或 task/domain router。

## 7. 创新性与工程贡献

本分支的创新点主要在系统贯通和可落地组合：

- 将 dataset、training policy、base model、action expert、structure policy 解耦，使所有组合都能进入统一训练/评估/后训练路径。
- 把 `T01-T28` 和 `S01-S32` 变成可交互选择的 policy matrix，而不是只在文档里列概念。
- 支持从 base pretrained 从头训练，也支持从已有 checkpoint 做后训练、failure replay、anti-forgetting、offline RL、DPO 和 distillation。
- 构建 H200 多卡训练 profile，保留显式大 batch，配合 NCCL/DeepSpeed/monitor/disk guard 提高长时间训练稳定性。
- 修复 CALVIN eval 的 normalization key、state dimension、FAST token decode、server/client 并发、GPU 残留和 checkpoint 损坏定位问题。
- 将 ensemble 变成标准 run 输出，可直接评估或继续后训练，同时通过结果证明异构参数 soup 的风险。

## 8. 未闭合技术问题

当前实验还留下几类需要继续验证的技术问题：

1. **评估方差问题**：GTY MoE 60k 和 WMH MoE95k LoRA Aug 的样本规模不完全一致，`n=100`、`n=300`、`n=1000` 之间不能简单横向排序。
2. **checkpoint averaging 边界**：异构 MoE/router soup 已经表现出明显退化，同构 checkpoint averaging 与异构参数平均需要分开评估。
3. **action-level ensemble**：当前结果只证明参数 soup 风险，并未否定以 value、uncertainty 或 task router 为权重的 action-level ensemble。
4. **failure-to-training 闭环**：CALVIN failure sequences 还可以转成 `T20/T24/T27` 的 hard-task replay、advantage-weighted BC 和 preference ranking。
5. **near miss 精度问题**：near miss 高的模型需要用 gripper/contact head、action-bound layer 和精度型后训练验证是否能转化为真实 success。
6. **world branch 作用边界**：Cosmos/world branch 更适合作为 future latent regularizer 还是直接 action backbone，需要在同等训练预算下继续对比。
