# StarVLA Training / Structure Policy 完整矩阵

本文档是 `interaction/config/policies.json` 中 Training Policy `T01-T28` 和 Structure Policy `S01-S32` 的中文完整说明。交互式训练中的输入顺序是：

```text
dataset -> training policy -> base model -> action expert -> structure policy
```

使用方式：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --training-policy T01 T02 T06 T07 T11 T22 \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --structure-policy S01 S04 S07 S15 S27
```

几个必须区分的概念：

- **Training Policy** 决定训练阶段、数据采样、归一化、loss、冻结/解冻策略、后训练策略。它回答“怎么训练”和“优化什么目标”。
- **Structure Policy** 决定模型结构、connector、action head、world/future branch、router、safety/output layer。它回答“模型内部怎么组织”。
- **base model** 是 VLM/world backbone，例如 `qwen3vl_4b_action`、`qwen35_2b`、`cosmos_predict2`。
- **action expert** 是动作生成模块，例如 `QwenOFT`、`QwenFast`、`QwenPI`、`QwenGR00T`、`CosmoPredict2PI`。
- `launch-supported` 表示已有原生训练入口或原生 framework/head；`runtime-supported` 表示通过共享 trainer runtime hook、metadata、loss weight、regularizer、deployment metadata 或后训练流程落地，能随 checkpoint 续训/后训练运行。

## 组合原则

从头训练一般至少包含：

```text
T01 T02 T07
S01 S04 S27
```

然后按 action expert 添加专用策略：

| 路线 | Training Policy | Structure Policy | 推荐 base/action expert |
|---|---|---|---|
| OFT continuous BC | `T06 T11 T22` | `S07 S15` | `qwen3vl_4b_action + QwenOFT` |
| FAST/action-token | `T04 T05 T11 T22` | `S02 S08 S15` | `qwen3vl_4b_action + QwenFast` |
| PI/GR00T flow | `T08 T11 T22` | `S10 S11 S15` | `qwen3vl_4b_action + QwenPI/QwenGR00T` |
| DiT/diffusion | `T09 T11 T22` | `S12 S15` | high-dimensional action expert |
| VLA+VLM co-train | `T12 T13` | `S03 S04 S23 S25` | Bridge/OXE/RoboTwin/RoboCasa |
| world/future auxiliary | `T15 T16 T23` | `S20 S21 S22 S31` | `cosmos_predict2 + CosmoPredict2PI/OFT/GR00T` |
| post-training | `T20 T21 T24 T27 T28` | `S26 S27 S28 S30` | 已训练 checkpoint 或 ensemble checkpoint |

后训练不要使用 `run-mode continue`。后训练是新 run 从已有权重初始化：

```bash
bash interaction/bin/starvla-interact.sh train \
  --dataset calvin_abc \
  --training-policy T20 T21 T24 T27 T28 \
  --base-model qwen3vl_4b_action \
  --action-expert QwenOFT \
  --structure-policy S01 S26 S27 S30 \
  --posttrain-from /path/to/checkpoints/steps_60000_pytorch_model.pt
```

## 表 1：Training Policy 完整清单

| ID | Training Policy | 训练阶段 | 训练 base-model + action-expert 的哪些部分 | 怎么训练 / 参数策略 | 具体策略设计 | 数学理论逻辑 / 目标函数 |
|---|---|---|---|---|---|---|
| T01 | Data Schema + Normalization Policy | 训练前全阶段 | 不直接训练参数；决定所有 `θ_B`、`θ_A` 的输入/输出尺度 | 为每个 embodiment 建 `μ_e`、`σ_e`、`min_e`、`max_e`、`q99_e`；state/action 按 group 归一化；gripper binary 单独处理 | CALVIN 用 7D、H=8；OXE 用 7D、H=16；RoboCasa GR1 用 29D、H=16；RoboTwin 用 14D、H=50。RoboCasa/RoboTwin 必须按 arm/hand/waist/gripper 分组 | 把 heterogeneous action space 映射到 normalized local coordinates：`a_tilde_g=(a_g-μ_e,g)/σ_e,g` 或 `a_tilde_g=2(a_g-min)/(max-min)-1`。训练最小化 `L(π(x), a_tilde)`，部署时 denorm，避免多机器人 action scale 破坏共享梯度 |
| T02 | Mixture Sampling Policy | 训练前 + SFT + co-train | 不直接训练参数；影响所有梯度采样分布 | 采样器参数：dataset temperature `α_d`、task temperature `α_t`、embodiment balance weight `β_e` | 建议 `p(D_i) ∝ n_i^α_d`，`α_d∈{0.3,0.5,0.7}`；同时按 task/embodiment 设置 minimum probability，防止 Fractal/OXE 淹没小数据集 | 经验风险变为重加权风险：`min_θ Σ_i p_i E_{D_i} L_i(θ)`。跨机器人数据有利于 generalist policy，但必须防止大数据源支配训练 |
| T03 | Robot-domain VLM-only Pre/Post-training | P0：action 训练前 | 训练 `θ_C`、`θ_ad`；可选训练 VLM top layers LoRA；冻结大部分 `θ_V`、`θ_L` | 用 robot images + language 做 caption、QA、object grounding、temporal ordering；入口为 `train_starvlm.py` | 数据来自 CALVIN/OXE/RoboCasa/RoboTwin image-language pairs；不输出动作，只让 VLM 熟悉 robot camera、object、task language | `L_VLM=-Σ_t log p_θB(y_t given y_<t,o,l)`，目标是让 robot-domain 视觉语言表示更适合后续动作训练 |
| T04 | Action-token Embedding Warmup | P0/P1：action-token 路线必做 | 训练 `θ_tok`：新增 action token embedding、LM head rows、action special token projection；冻结 `θ_L` 大部分 | 先不训完整 action expert，只让 action token embedding 与连续动作 tokenizer 对齐 | 对 Qwen3.5-*-Action、Qwen3-VL-4B-Instruct-Action 必做；当前 action models 本质是 embedding resize，不是已训练 action-policy checkpoint | 若 `z=Tok(a)`，目标为 `L_CE=-Σ_i log p_θtok(z_i given c,z_<i)`，让离散 action token 有稳定几何语义 |
| T05 | FAST / DCT Action Tokenizer Training | P1：action tokenizer 预训练 | 训练 `Tok_φ`、`Dec_ψ`；可训练 `θ_FAST` token prediction head；冻结 `θ_B` | 对 action chunk 做 DCT/频域压缩、量化、token 化，再训练自回归 token prediction | 高频 dexterous/long-horizon action 优先用 FAST/DCT，而不是逐维逐时间 binning，以降低 token 长度和训练成本 | `u=DCT(a_0:H-1)`，`z_i=Q(u_i)`；训练 tokenizer `L_tok=CE(z,z_hat)+λ_rec ∥a-Dec(z)∥_1` |
| T06 | OFT / MLP Continuous BC Baseline | SFT 主 baseline | 冻结 `θ_V`、`θ_L`；训练 `θ_C`、`θ_E`、`θ_MLP`、`θ_G`；可选 top-layer LoRA `θ_ad` | continuous action chunk regression；gripper 单独 BCE；先 head-only，再 projector，再 LoRA | CALVIN、RoboTwin、Bridge 第一强 baseline。OpenVLA-OFT 使用 parallel decoding、action chunking、continuous actions、L1 objective | `π_θ(x)=Denorm_e(g_θMLP(c))`；`L=Σ_g w_g ∥a_tilde_g-a_hat_g∥_1 + λ_g BCE(g,g_hat)` |
| T07 | Action Chunking + Horizon Curriculum | SFT 全阶段 | 训练当前选定 action head；不改变 base 参数策略 | 先短 horizon，再逐步增加到数据集默认 horizon；部署用 receding horizon | CALVIN: 4→8；OXE: 8→16；RoboCasa: 8→16；RoboTwin: 16→32→50。ACT 说明 action chunking 可降低 compounding error | 训练 `π(a_t:t+H-1 given o_t,s_t,l)`，部署每次执行前 k 步再重规划 |
| T08 | GR00T / PI Flow-Matching Action Expert Training | SFT / foundation co-train | 训练 `θ_FM`、`θ_E`、`θ_C`；PI 额外训练 layerwise fusion；可选 LoRA `θ_ad` | 冻结大部分 VLM；以 VLM hidden states 条件化 flow action expert | 适合 RoboCasa GR1 29D、RoboTwin 14D long horizon、跨 embodiment；GR00T/π0 路线使用 VLM System-2 + flow/diffusion-like System-1 | 采样 `x_0~N(0,I)`、`x_1=a_tilde`、`τ~U(0,1)`；`x_τ=(1-τ)x_0+τx_1`，目标速度 `u=x_1-x_0`；`L_FM=E ∥v_θ(x_τ,τ,c)-u∥^2` |
| T09 | DiT / Diffusion Action Expert Training | SFT / high-dimensional action | 训练 `θ_DiT`、`θ_C`、`θ_E`；冻结 `θ_B` 或只开 LoRA | 把 action chunk 当作 diffusion sample，条件为 VLM/state tokens | Diffusion Policy 建模 conditional denoising diffusion；RDT-1B 用 diffusion transformer 做多机器人双臂 manipulation foundation | 前向噪声 `x_τ=α_τ a_tilde+σ_τ ε`；噪声预测 `L_diff=E ∥ε-ε_θ(x_τ,τ,c)∥^2` 或 v-pred |
| T10 | Hybrid Token + Continuous Residual Training | SFT 中后期 | 同时训练 `θ_FAST` + `θ_MLP/θ_FM` + `θ_G`；冻结或 LoRA `θ_B` | coarse action token 预测 + continuous residual 修正 | 对 token 路线减少 quantization loss；对 continuous 路线增加 mode/skill 先验 | `z=Tok(a)`，`a_hat=Dec(z_hat)+r_θ(c,z_hat)`；`L=λ_CE L_CE+λ_1 ∥a-a_hat∥_1` |
| T11 | Grouped Action + Separate Gripper Training | SFT 全阶段 | 训练 `θ_G`：group heads、gripper head、action-bound head | action 按 body group 分头；gripper 用 BCE/focal；arm/hand/waist 用 L1/FM/diffusion | RoboCasa GR1 分 left_arm/right_arm/left_hand/right_hand/waist；RoboTwin 分 left/right arm + gripper | 共享 trunk 下条件独立分组：`p(a given c)=Π_g p(a_g given c,e,g)`；`L=Σ_g w_g L_g + λ_grip L_BCE` |
| T12 | Multi-objective VLA + VLM Co-training | Co-train | 训练 `θ_C`、`θ_E`、`θ_A`、`θ_ad`；可选训练 VLM top layers | 入口 `train_starvla_cotrain.py`；动作 loss + VLM/grounding/future/inverse losses | Bridge+Fractal、RoboCasa、RoboTwin 适用；保持 VLM 语义能力，防止只学动作回归 | `L=λ_a L_action+λ_v L_VLM+λ_g L_ground+λ_inv L_inv+λ_f L_future` |
| T13 | Cross-Embodiment Co-training | Foundation SFT | 训练 `θ_E`、`θ_C`、`θ_A`；可选 `θ_ad`；`θ_B` 多数冻结 | 多机器人 mixture；每个 embodiment 有 normalization、action mask、embodiment token、group head | Bridge/Fractal 7D、RoboCasa 29D、RoboTwin 14D 不能强行同构；用 shared trunk + embodiment-specific output | `L=Σ_e p(e) E_D_e [M_e * L(π_θ(x,e),a)]`，`M_e` 是 action dimension mask |
| T14 | Representation Alignment / REPA-style Training | P1 / co-train | 训练 `θ_C`、`θ_A` 中的 projection；可选 top-layer LoRA；teacher frozen | 用 frozen visual/video foundation model 的 representation 对齐 action expert hidden states | 对 DiT/FM action expert 尤其有效；REPA 通过把 diffusion hidden states 对齐到外部视觉表征提高训练效率 | teacher `φ(o)`、action hidden `h_θ`；`L_align=∥P h_θ - sg(φ(o))∥_2^2` |
| T15 | FRAPPE / Future Representation Alignment | P1 / co-train / post-train | 训练 `θ_W`、`θ_C`、`θ_A`；teacher VFM frozen | 预测多个未来 observation latent，不要求推理时生成未来图像 | 将 world modeling 注入 generalist policy，避免依赖像素重建和测试时未来观测 | `z^T_t+k=sg(φ(o_t+k))`，`z_hat_t+k=g_θW(c_t,a_t:t+k)`；`L_future=Σ_k ∥z_hat_t+k-z^T_t+k∥^2` |
| T16 | Fast-WAM Train-time World Co-training | Co-train / post-train | 训练 `θ_W`、`θ_C`、`θ_A`；推理期可丢弃 `θ_W` | 训练时视频/未来 latent co-training；测试时只用 action expert | 保留 video/world co-training 收益，但移除测试时显式 future imagination，降低延迟 | `L=L_action+λ_w L_world`；推理 `a=π_θA(c)`，不执行未来生成 |
| T17 | DIAL Latent Intent Bottleneck Training | P1 warmup → joint SFT | 训练 `θ_W` latent intent head、`θ_A` inverse/action decoder、`θ_C`；分阶段解冻 `θ_ad` | 两阶段：先训 latent future/intent，再端到端 joint | 用 latent intent bottleneck 解耦高层 VLM intent 与低层 action；System-2 预测 latent foresight，System-1 inverse dynamics 解码动作 | `z_g=g_θW(c)`，`a=π_θA(s,z_g)`；warmup `L_z=∥z_g-sg(z_future)∥^2`，joint `L=L_action+λ_z L_z` |
| T18 | Human/Synthetic Video Pretraining + Retargeting | P0/P1 | 训练 `θ_C`、`θ_W`、`θ_A` 的 latent action / inverse dynamics；不一定训练 full VLM | 从 human egocentric/synthetic video 学 affordance、contact、latent skill，再 retarget 到 robot actions | 适合 humanoid/GR1、bimanual；RoboCasa/RoboTwin 可混 synthetic trajectories | 无 action 视频先学 latent action `z_a=q(o_t,o_t+k)`，再学 `p(o_t+k given o_t,z_a)` 和 robot inverse dynamics |
| T19 | Uncertainty / Probabilistic BC Training | SFT 中后期 | 训练 `θ_Q` 或 action head 的 `μ,σ` 输出；action trunk 可共享 | 不只预测 action mean，还预测 variance/mixture weight | 用于 OOD detection、RL exploration、safety shield、trajectory ranking | Gaussian NLL：`L_NLL=Σ_i ((a_i-μ_i)^2/(2σ_i^2)+log σ_i)`；mixture 用 `-log Σ_k π_k N(a;μ_k,σ_k)` |
| T20 | Hard-task Curriculum + Failure Replay | Post-train | 训练当前 `θ_A`、`θ_C`、`θ_ad`；`θ_B` 多数冻结 | 从 rollout/eval 中收集失败任务，按 failure rate 上采样 | 对 CALVIN long-horizon、RoboTwin randomized、RoboCasa novel split 有用 | `p_i ∝ (ε+fail_i)^γ`；优化 `Σ_i p_i L_i`，把梯度预算分配给高失败率任务 |
| T21 | Anti-forgetting Post-training | Post-train / dataset adaptation | 训练 `θ_A`、`θ_C`、`θ_ad`；冻结 `θ_B` lower layers；保留 teacher `π_0` | 新数据训练时混 replay、KL to base、hidden distillation、视觉语义 replay | 从 OXE 到 RoboCasa/RoboTwin 或 CALVIN 到 OXE 时必须考虑 | `L=L_new+λ_replay L_old+λ_KL KL(π_θ ∥ π_0)+λ_h ∥h_θ-h_0∥^2` |
| T22 | Visual/Temporal/Language/Action Augmentation Training | P0/SFT/co-train | 不新增参数；影响所有 `θ` | 图像 crop/color/camera dropout；temporal crop/frame skip；instruction paraphrase/dropout；action noise/smoothing | RoboTwin randomized、RoboCasa sim、OXE real 数据启用不同强度 | invariant risk：`min_θ E_T L(f_θ(Tx),y)`；增强必须 preserve action label，不能做破坏因果的 time reversal |
| T23 | Sim-to-Real / Domain Randomization Policy | P1 / post-train | 训练 `θ_C`、`θ_A`、`θ_ad`；可加 domain adapter | synthetic/sim 大比例预训，real/clean 数据 anchor replay；domain token + adapter | RoboCasa/RoboTwin sim-heavy；Randomized split 用于 OOD robustness | `L=E_sim w_sim L + E_real w_real L + λ MMD(h_sim,h_real)`，domain randomization 扩展训练分布，real replay 防止 sim bias |
| T24 | Offline RL / Advantage-weighted BC | SFT 后 | 训练 `θ_A`、`θ_Q`；可训练 adapters；冻结大部分 `θ_B` | 从 success/failure、reward、return 估计 advantage；用 BC-regularized policy improvement | 没有在线环境时优先；可从 CALVIN/RoboCasa/RoboTwin success signal 生成 | `L=-E[w(s,a) log π_θ(a given s)] + β KL(π_θ ∥ π_BC)`，`w=exp(A/η)` 或 clipped advantage weight |
| T25 | Online Sim RL with BC/KL Regularization | SFT 后 RL | 训练 `θ_A`、`θ_Q`、`θ_ad`；冻结 `θ_V`、`θ_L`；必要时只训 action head | 在 RoboCasa/RoboTwin/CALVIN sim rollout；PPO/SAC/TD-MPC2；强 KL 到 SFT policy | TD-MPC2 在 latent world model 中做 trajectory optimization，适合连续控制后训练 | `max_θ J=E_π Σ γ^t r_t`，约束目标 `L_RL=-J+β KL(π_θ ∥ π_SFT)+λ_BC L_BC` |
| T26 | Flow-based Online RL / π-StepNFT-style | SFT 后 RL，针对 FM head | 训练 `θ_FM`；可训练 `θ_Q/success`；冻结 VLM | 对 flow ODE step-wise trajectory 做成功/失败 guidance；不依赖完整 likelihood | 面向 flow-based VLA 的 critic-free、likelihood-free、step-wise negative-aware online RL fine-tuning | 对 flow 轨迹 `x_τ` 施加方向性更新；抽象目标 `L=L_FM+λ rank(v_θ;τ^+,τ^-)+β KL_path`，优化生成路径而非单步 action |
| T27 | Preference / DPO-style Trajectory Ranking | Post-train / RLHF/RLAIF | 训练 `θ_A`、`θ_Q`、`θ_ad`；base frozen | 用 success trajectory vs failure trajectory，或 VLM/world-model judge 给 pairwise preference | 无 dense reward 时可用；适合 long-horizon task | `L_DPO=-log σ(β((logπ_θ(τ+)-logπ_θ(τ-))-(logπ_0(τ+)-logπ_0(τ-))))`，用 reference policy 防止漂移 |
| T28 | Final Distillation / EMA / Checkpoint Averaging | Finalize / deployment | 学生模型训练 `θ_A^S`、`θ_C^S`；teacher 是 top-k/ensemble | 多 checkpoint ensemble → single deploy model；EMA/SWA；sampler distillation | 默认 top-k 按 loss 保存；建议结合 eval-success top-k | teacher distribution `p_T(a given x)=Σ_m w_m p_m(a given x)`；学生最小化 `KL(p_T ∥ p_S)` 或 `∥a_T-a_S∥` |

## 表 2：Structure Policy 完整清单

| ID | Structure Policy | 改进哪一个 structure | 使用什么改进 | 放在什么地方 | 数学理论原理 / 结构逻辑 |
|---|---|---|---|---|---|
| S01 | Base VLM Preservation + PEFT Structure | `θ_B = θ_V + θ_L` | 冻结 lower/mid layers；top layers 加 LoRA/adapter；vision backbone 默认不全量改 | Qwen3.5/Qwen3-VL/Cosmos backbone 内部 transformer blocks，优先 top-k layers | `W'=W+ΔW`，LoRA `ΔW=BA`，`rank(BA)=r<<d`。用低秩子空间适配 robot domain，降低 catastrophic forgetting 与显存成本 |
| S02 | Action-token Embedding Structure | tokenizer + embedding table + LM head | 新增 action token embedding、action special tokens、action-token LM head rows | Qwen3.5-Action / Qwen3-VL-Action token embedding 与输出 head | 当前 action models 主要是 tokenizer/embedding 扩展，需要学习离散 action code `z` 的 embedding，使 `p(z given c)` 与动作空间几何对齐 |
| S03 | Multi-camera Connector | vision connector `θ_C` | per-camera projector + camera token + cross-view attention / Perceiver resampler | vision tokens 进入 VLM 前，或 VLM hidden 到 action expert 前 | 多相机输入 `V_i` 加 camera embedding 后融合：`Z=Attn(Q,K=[V1...Vn],V=[V1...Vn])`；RoboTwin 三路相机必须显式 camera identity |
| S04 | Embodiment / FPS / Control-mode Token Structure | input token structure | 加 `embodiment`、`fps`、`control_mode`、`horizon`、`camera` tokens | text prompt 前缀、state embedding 前缀、action expert condition | `c=f(o,s,l)+E_emb(e)+E_fps(Δt)+E_mode(m)`；避免不同 FPS 与 abs/delta/qpos 控制语义混叠 |
| S05 | State / Proprio Encoder | proprio branch | MLP/Transformer state encoder；state group embedding；sin/cos joint encoding | VLM hidden 与 action expert condition 之间 | `h_s=Enc_s([s_t-k:t, group_id])`，再 `c'=CrossAttn(c,h_s)`；high-DoF GR1/RoboTwin 必须显式编码关节闭环状态 |
| S06 | Temporal Memory Block | temporal structure | history-frame transformer、action-history token、sliding memory token | vision-language backbone 之后、action expert 之前 | `M_t=Transformer([φ(o_t-k:t),s_t-k:t,a_t-k:t-1])`；把 POMDP 历史压缩为 belief state `b_t≈p(s_env_t given history)` |
| S07 | OFT / MLP Action Head Upgrade | `MLP_ActionHeader.py` / OFT head | SwiGLU MLP、residual MLP、grouped output、uncertainty output、action-bound output | action special token hidden state 后 | `a_hat=Bound(MLP(LN(h_action)))`；residual `a_hat=a_base+r_θ(h)`。continuous action + L1 + chunking 是 OFT 路线核心 |
| S08 | FAST Token Action Head | `fast_ActionHeader.py` | DCT/FAST tokenizer、parallel token decoding、detokenization consistency | action token prediction head；LM head 或独立 token head | `z=Q(DCT(a))`，训练 `p(z given c)`；短 token 序列降低长 horizon 自回归成本 |
| S09 | Hybrid Coarse-token + Continuous Residual Head | action representation head | FAST/VQ coarse token head + residual continuous head | action expert 最后一层，token branch 与 regression/FM branch 并联 | `a_hat=Dec(z_hat)+r_θ(c,z_hat)`；离散 token 表达 mode/skill，continuous residual 保精度 |
| S10 | GR00T / Flow-matching Action Expert | `GR00T_ActionHeader.py` | DiT-B/L action expert、rectified flow、condition cross-attention、flow time embedding | VLM hidden/state tokens → action flow decoder | `dx/dτ=v_θ(x,τ,c)`；训练 `L=E ∥v_θ(x_τ,τ,c)-(x_1-x_0)∥^2` |
| S11 | PI Layerwise Cross-DiT Head | `LayerwiseFM_ActionHeader.py` | 多层 hidden states gated fusion，而非只用最后层 | VLM backbone selected layers 到 flow action expert condition path | `c=Σ_l α_l P_l h_l`，`α=softmax(g(e,task,s))`；低层保空间纹理，高层保语义任务，不同 embodiment 自适应选层 |
| S12 | DiT Diffusion Action Head | `DiTActionHeader.py` | Diffusion transformer、v-pred、classifier-free condition dropout、min-SNR weighting | action expert 独立 DiT block | diffusion head 学 denoising/score field：`∇_a log p(a given c)`；适合多峰动作分布 |
| S13 | Bimanual Cross-attention Decoder | RoboTwin action head | 左右臂独立 head + cross-attention coordination token | shared action trunk 后，输出 left/right joints/grippers 前 | `h_L'=Attn(h_L,h_R,h_R)`，`h_R'=Attn(h_R,h_L,h_L)`；建模双臂协同 `p(a)=p(a_L,a_R given c)` |
| S14 | GR1 / Humanoid Grouped Decoder | RoboCasa GR1 action head | arm/hand/waist 分组 decoder + coupling block + waist stabilization head | shared trunk 后，输出 29D action 前 | `p(a)=p(waist)p(arms given waist)p(hands given arms)`；稳定 high-DoF humanoid 输出 |
| S15 | Separate Gripper / Contact Head | gripper/contact structure | gripper BCE head、contact state head、grasp phase token | action head 并联分支 | `p(g=1 given c)=sigmoid(h_g)`；gripper/contact 是离散或相位变量，不应与连续 joint 用同一 L1 处理 |
| S16 | Hierarchical Action MoE | action expert FFN / grouped decoder | shared expert + embodiment expert + action-group expert + task expert | 优先放在 action head / adapter / connector，不先改全 backbone | `y=Σ_i r_i(x,e,g) E_i(x)`，`r=softmax(R(x,e,g))`；加 load-balance loss `L_lb=KΣ_i f_i p_i` |
| S17 | Adapter-MoE / Domain Adapter | LoRA/adapter structure | 每 dataset/domain/embodiment 一个 adapter，router 选择或加权 | VLM top layers、connector、action expert condition layers | `h'=h+Σ_i r_i A_i(h)`；比 full MoE 更稳，新机器人只加 adapter，不污染 shared backbone |
| S18 | MoT Fast-Slow Dual Transformer | base/action dual-system | slow VLM System-2 + fast action transformer System-1 | slow side 处理图像/语言/语义；fast side 处理 proprio/history/action | `z_intent=F_s(o,l)` 低频更新，`a_t=F_f(z_intent,s_t-k:t,o_t)` 高频更新 |
| S19 | GR00T-like Dual-system Humanoid Architecture | humanoid VLA whole structure | VLM System-2 + diffusion/flow action expert System-1 | base model 与 action expert 解耦，通过 latent tokens 通信 | `p(a given o,l,s)=∫ p_A(a given z,s)p_B(z given o,l) dz`；适合 humanoid high-DoF 控制 |
| S20 | Latent Intent Bottleneck | world/intent/action interface | `z_goal` / `z_intent` bottleneck token；future latent predictor；inverse dynamics decoder | VLM hidden 与 action expert 中间 | 信息瓶颈：`z` 保留任务相关未来信息、丢弃视觉冗余，降低端到端优化难度 |
| S21 | World / Future Latent Head | WAM auxiliary structure | future latent prediction head、inverse dynamics head、success prediction head | action expert 旁路；训练期启用，部署可关闭 | `z_hat_t+k=F_W(z_t,a_t:t+k)`，`a_hat=F_inv(z_t,z_t+k)`；训练期 world regularizer，推理期可移除 |
| S22 | Future Representation Alignment Head | representation alignment branch | 多未来 horizon representation heads；teacher VFM projectors | VLM/action hidden 到 frozen VFM latent 的 projection | `L=Σ_k ∥P_k h_t - sg(φ(o_t+k))∥^2`；对齐未来表征，提升策略时序理解 |
| S23 | Object-slot / Affordance Structure | visual-semantic structure | object slots、affordance heatmap head、object-action cross-attention | vision connector 后，action expert condition 前 | `slots=SlotAttn(V)`，`a=π(c,slots)`；affordance `A(u,v)=p(contact at pixel)` 使动作条件显式关注可操作物体 |
| S24 | Spatial / 3D Connector | vision connector / state grounding | depth/RGB-D branch、point tokens、Ego3D position encoding、adaptive action grids | vision encoder 到 VLM/action expert 之间 | 点或 patch 加 3D 位置编码 `p_i=(x_i,y_i,z_i)`；适合抓取、放置、遮挡和相机变化 |
| S25 | Qwen-style Dynamic Resolution + Time Encoding Adaptation | base VLM input structure | dynamic resolution image tokens、absolute time encoding、video-frame time tokens | vision encoder/preprocessor 与 multimodal positional embedding | `PE(t)=PE_abs(frame_time/FPS)`；避免不同 FPS 数据时间混叠，保留机器人视频时序 |
| S26 | Value / Success / Reward Head | auxiliary head `θ_Q` | value head、success classifier、progress head、failure classifier | action trunk hidden 后，并联输出 | `V(c)=E[R given c]`；为 offline RL、DPO ranking、failure replay、早停和 checkpoint selection 提供信号 |
| S27 | Safety Projection / Action-bound Layer | output layer | tanh bound、joint-limit projection、velocity/jerk shield、workspace constraint layer | final action output 后，发送 robot controller 前 | `a=low+(high-low)(tanh(a_raw)+1)/2`；更强安全层 `a_safe=argmin_u ∥u-a_raw∥^2` subject to constraints |
| S28 | Uncertainty / Ensemble Head | action output distribution | variance head、mixture density head、ensemble logits、OOD score | action head 并联；可与 value/safety 共用 trunk | `p(a given c)=Σ_k π_k N(a;μ_k,Σ_k)`；输出不确定性用于 OOD、safety 和 trajectory ranking |
| S29 | Flow / Diffusion Sampler Acceleration | FM/DiT inference structure | adaptive ODE steps、one-step student、consistency distillation | action expert sampler，不改 VLM | flow/diffusion 推理慢；用 adaptive steps 或 student distillation：`L=∥a_student-a_teacher∥` |
| S30 | Policy Ensemble / Router Structure | deployment policy structure | 多 head ensemble、dataset/embodiment router、eval-score weighted merge | action expert 层或 checkpoint ensemble 层 | `π(a given x)=Σ_m w_m(x) π_m(a given x)`；用 eval-score 或 router 决定权重 |
| S31 | Cosmos / Video2World Auxiliary Interface | world model interface | frozen Cosmos latent teacher、synthetic video/data generator、policy evaluator | StarVLA `θ_W` teacher branch 或 data augmentation pipeline | Cosmos-Predict2 不建议直接替代低延迟 action backbone，更适合作为 `φ_world(o,a)` teacher 或 synthetic data engine |
| S32 | Cross-modal Speech / Omni Input Extension | input modality structure | speech instruction encoder、audio-token adapter、text-speech shared instruction embedding | language input 前端，进入 VLM prompt/embedding | 面向 humanoid natural interaction 的中后期扩展；语音/音频进入 instruction embedding，不应优先于 action expert 收敛 |
