# SII SummerCamp StarVLA 技术总结报告

本文按《创智夏季营 · 具身智能方向实训考题》的交付要求整理：代码工程实现、技术路线论证、CALVIN ABC→D 测评结果、Failure Pattern 分析，以及个人贡献说明。赛题核心是在 StarVLA 框架上实现 Action Policy Model，并在 CALVIN ABC→D 上评估 Task 1 到 Task 5 成功率和 Average Chain Length。

## 1. 工程实现

本次实现的重点不是只训练单个模型，而是把 StarVLA 整理成可复现、可组合、可恢复的训练与评估系统。整体流程拆成五个轴：

```text
datasets -> training policy -> base model -> action expert -> structure policy
```

其中 `datasets` 负责 action/state/horizon、数据路径、data mix 和归一化配置；`training policy` 负责训练目标、采样、后训练、RL/偏好优化、蒸馏等策略；`base model` 负责视觉语言或世界模型骨干；`action expert` 负责连续动作、离散 action token、flow/diffusion 或 MoE 动作头；`structure policy` 负责 LoRA/adapter、状态编码、多相机、时序记忆、MoE、value/safety/uncertainty head 等结构增量。

已支持的数据集包括 CALVIN、LIBERO、Open X/Bridge/RT-1 mixture、RoboTwin、RoboCasa GR1、RoboCasa365 smoke 和 robot image-language VLM-only 数据。已支持的 base model 包括 Qwen3.5 0.8B/2B/4B/9B 及 Action-token 版本、Qwen3-VL-4B Instruct 及 Action-token 版本、Cosmos-Predict2-2B Video2World。已支持的 action expert 覆盖 FAST、OFT、PI、GR00T、Adapter、Dual、ABot_M0、LangForce、QwenPI/QwenOFT/QwenFast/QwenGR00T/QwenDual、CosmoPredict2PI/OFT/GR00T、WanPI/OFT/GR00T、Gemma4PI/GR00T、CosmosGR00T、InternVLA-M1 等组合。

训练策略侧实现并登记了 `T01` 到 `T28` 的完整 policy 清单，覆盖数据 schema/normalization、mixture sampling、VLM-only robot domain 预训练、action-token warmup、FAST/DCT tokenizer、OFT continuous BC、action chunking curriculum、flow/diffusion action expert、hybrid token + residual、grouped gripper、VLA+VLM co-training、cross-embodiment co-training、future/world representation alignment、failure replay、anti-forgetting、offline RL、online RL、DPO-style trajectory ranking、EMA/SWA/checkpoint averaging。结构侧实现并登记了 `S01` 到 `S32` 的完整 policy 清单，覆盖 base VLM preservation + PEFT、action token embedding、多相机 connector、embodiment/FPS/control-mode token、state encoder、temporal memory、OFT/FAST/Flow/DiT action head、bimanual/humanoid grouped decoder、gripper/contact head、MoE/domain adapter、fast-slow dual transformer、latent intent、world/future latent head、spatial/3D connector、value/safety/uncertainty head、sampler acceleration 和 policy ensemble/router。

交互式系统方面，`interaction/bin/starvla-interact.sh` 现在是唯一主入口。它支持 catalog、环境检查、H200/NCCL 性能检查、训练 dry-run、tmux 管理训练、后训练、CALVIN websocket 评估、ensemble、日志监控、GPU 清理和磁盘守护恢复训练。训练输出统一进入 `/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/<experiment_group>/<run_id>/`，评估运行统一进入 `/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/`。这样训练配置、checkpoint、完整 accelerate state、top-k checkpoint、dataset statistics 和评估日志都能定位到同一套目录。

工程稳定性方面，主要做了以下修复和增强：

- 训练监控使用 Rich live dashboard 或 ANSI in-place fallback，避免持续刷屏；同时展示 step、epoch、loss、top-k、GPU 显存/功率/util、最近错误和 root-cause pattern。
- H200 训练启用 BF16、DeepSpeed ZeRO-2、NCCL async/nonblocking、NVLink P2P、较大通信 bucket、prefetch/persistent dataloader worker，并保留用户指定大 batch，不再自动退化到小 batch。
- 磁盘空间不稳定时通过 `starvla-disk-guard-resume.sh` 自动停训、等待空间恢复、从完整 state 续训，避免写坏 checkpoint。
- 修复 CUDA OOM、残留 accelerate 进程、tmux 多次启动导致的 GPU 进程堆叠问题，提供 GPU orphan 清理脚本。
- 修复 CALVIN eval 中 `unnorm_key=franka` 与 checkpoint 只有 `new_embodiment` 的不一致问题，服务端和客户端均可自动映射到唯一可用 normalization key。
- 修复 FAST eval 中 action token 为 `None` 时反复刷错误的问题，使错误能在服务端明确定位，而不是在 client 侧表现为 actions missing。
- 为 ensemble 产物生成普通 StarVLA run 结构，包含 `config.yaml`、`dataset_statistics.json`、`ensemble.json` 和 `checkpoints/steps_ensemble_pytorch_model.pt`，可直接进入评估或后训练。

## 2. 世界模型路线尝试

赛题允许选择 VLA 或 World Model 路线。本次除 Qwen-VLA 路线外，也尝试了 Cosmos-Predict2 世界模型路径。核心思路是把 Cosmos-Predict2 这种 Video2World DiT 作为视觉时序表征骨干或世界模型 teacher，再接入 PI/GR00T/OFT action expert。与纯 VLA 直接学习 `p(a | o, l, s)` 不同，世界模型路线显式利用视频生成/未来状态预测能力，使模型先学习环境动态、物体运动和任务时序，再把这些 latent condition 输入动作头。理论上它更适合长程规划和跨环境泛化，因为动作不是只依赖当前观测，而是被未来 latent 或 world representation 约束。

本地训练的 Cosmos-Predict2-PI run：

```text
/inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/checkpoints/cosmos_predict2-CosmoPredict2PI-calvin_task_ABC/interactive_calvin_0519_093049
```

该 run 中完整 checkpoint 包括 `steps_4800`、`steps_5300`、`steps_5500`，其中 `steps_5500_pytorch_model.pt` 约 17.3GB，是 top-k 中 loss 最低的完整权重；`steps_5900_pytorch_model.pt` 只有 4MB，判断为中断产生的不完整文件，不用于评估或提交。`steps_5500` 的关键训练指标为：

```text
loss / action_dit_loss: 0.231206
mse_score: 0.006678
global_batch_size: 512
epoch: 2.6356 / total_epochs: 3.5817
timing/model: 4.18s per logged step
```

公共评估日志中也有 Cosmos-Predict2-GR00T full fine-tuning 的 CALVIN prefix-first100 小规模结果：

| checkpoint | n | avg chain | Task1 | Task2 | Task3 | Task4 | Task5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| steps_10000 | 未记录 | 0.367 | 33.3% | 3.3% | 0.0% | 0.0% | 0.0% |
| steps_12000 | 100 | 0.610 | 45.0% | 12.0% | 3.0% | 1.0% | 0.0% |
| steps_18000 | 100 | 0.580 | 42.0% | 12.0% | 3.0% | 1.0% | 0.0% |
| steps_24000 | 100 | 0.500 | 41.0% | 9.0% | 0.0% | 0.0% | 0.0% |

分析上，世界模型路线在短训练下已经能学到部分单步任务，Task1 成功率可到 40% 以上，但 Task4/Task5 基本没有形成稳定长程能力。主要原因有三点：第一，世界模型骨干推理和训练开销高，在 2 天赛题时限下同等 GPU 时间内训练步数少；第二，CALVIN 的动作精度和反归一化非常敏感，世界模型 latent 对物体未来状态有帮助，但仍需要更长的 action expert 后训练才能把 latent 转成稳定控制；第三，prefix-first100 小样本主要反映功能可行性，不能替代正式 n1000 结果。因此世界模型路线作为探索和架构验证成立，但最终竞争成绩仍优先采用训练更充分、评估更稳定的 Qwen/MoE VLA 路线。

## 3. 小模型与小 epoch 训练尝试

在时间和磁盘都受限的情况下，先做小模型、小 steps、小 batch 或短周期训练，用于验证 pipeline、定位数据/归一化/评估问题。这部分的价值不是直接冲最终分数，而是快速排除系统性错误，例如 action_dim/state_dim 不匹配、CALVIN data mix 未注册、FAST token decode 失败、评估 unnorm key 错误、server/client 并发数不适配等。

尝试过的方向包括：

- Qwen3.5/Qwen3-VL 小尺寸或 action-token 版本配合 OFT/FAST，验证 base model 与 action expert 解耦是否能跑通。
- 低 steps CALVIN 训练，优先保存 top-k checkpoint 和完整 accelerate state，观察 loss、mse_score、epoch/step、GPU 利用率是否正常。
- 小 batch 或保守 throughput profile，用来确认数据读取、tmux orchestration、websocket eval 和 checkpoint 保存逻辑。
- 8k/15k 级别训练的 MoE/connector/augmented 数据变体，用较短时间判断泛化趋势。

代表性评估结果：

| 模型或设置 | n | avg chain | Task1 | Task2 | Task3 | Task4 | Task5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WMH base8k | 300 | 1.050 | 54.0% | 25.7% | 13.7% | 8.0% | 3.7% |
| WMH state8 connector steps8000 fast w4x8 | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% |
| WMH augmented hard v2 | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% |
| 本地普通 checkpoint eval | 1000 | 0.081 | 6.8% | 1.3% | 0.0% | 0.0% | 0.0% |

从泛化性角度看，小 epoch 训练可以很快提升 Task1/Task2，但 Task3 到 Task5 对长程误差累积、动作精度和跨环境泛化要求更高。单纯降低 batch 或训练步数通常会导致模型只学到局部 affordance，例如开抽屉、移动 slider、turn on LED 等容易任务，而 rotate、lift、stack、place 等需要更精细闭环的任务仍然失败。增强数据和 MoE/connector 结构带来的提升更明显，说明 ABC→D 的关键不是只增加参数量，而是要处理 domain shift、任务多样性和 action mode 多峰性。

## 4. 最优模型 ensemble 尝试

本次在测评前尝试了 checkpoint soup ensemble，希望利用不同训练 run 的互补性提升稳定性。候选模型包括：

1. GTY MoE 60k：`steps_60000_pytorch_model.pt`
2. GTY MoE posttrain 95k：`steps_95000_pytorch_model.pt`
3. WMH adaptive MoE 15k：`steps_15000_pytorch_model.pt`
4. WMH adaptive MoE 15k final model：`final_model/pytorch_model.pt`

已知单模型结果：

| 模型 | n | avg chain | Task1 | Task2 | Task3 | Task4 | Task5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GTY MoE 60k | 100 | 1.910 | 76.0% | 54.0% | 32.0% | 17.0% | 12.0% |
| GTY MoE posttrain 95k | 300 | 1.640 | 72.7% | 42.0% | 24.0% | 15.7% | 9.7% |
| WMH adaptive MoE 15k | 1000 | 0.910 | 50.9% | 23.4% | 10.4% | 4.4% | 1.9% |
| WMH state8 connector 8k | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% |
| WMH augmented hard v2 | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% |

实现上，`interaction/starvla.py ensemble --recipe calvin_ultimate_moe` 会读取多个 checkpoint，检查共同参数 key 和 tensor shape，然后按权重做 weighted soup。默认权重为 `0.38,0.42,0.10,0.10`，偏向两个 GTY 模型，同时保留 WMH 模型作为多样性来源。输出目录包含标准 StarVLA eval 所需的 `config.yaml`、`dataset_statistics.json` 和 `checkpoints/steps_ensemble_pytorch_model.pt`。

ensemble 评估结果：

```text
run: /inspire/qb-ilm2/project/26summer-camp-10/26220447/data/starvla/runs/20260520_054803_starvla_eval_steps_ensemble_pytorch_model
n: 1000
avg chain: 0.302
Task1: 25.8%
Task2: 3.8%
Task3: 0.6%
Task4: 0.0%
Task5: 0.0%
```

结果没有超过最优单模型，反而明显下降。原因判断是：checkpoint soup 适合同一初始化、同一架构、相近训练轨迹、参数处于同一 basin 的模型；但当前四个候选包含不同 MoE/router、不同训练阶段、不同数据增强和可能不同 adapter/normalization 细节。对 MoE 结构来说，直接平均 router/expert 参数可能破坏专家分工；对 action head 来说，参数平均也可能把多个动作 mode 平滑掉，导致长程任务更不稳定。因此本次 ensemble 的工程实现是可落地的，但在当前候选集合上不适合作为最终提交模型。

更合理的后续 ensemble 方式是 action-level ensemble 或 eval-time router：每个模型独立 forward，按任务类型、历史成功率、uncertainty 或 value head 选择/加权 action，而不是直接平均参数。对于 CALVIN 这种长程任务，也可以只对同构 GTY 60k/95k 做小范围 soup，再配合 failure-aware post-training，而不是把异构 WMH MoE 同时平均进同一权重文件。

## 5. Failure Pattern 分析

根据评估日志和 task-level success，主要失败模式如下：

- 误差累积：Task1/Task2 已经有一定成功率，但 Task3 到 Task5 快速下降。根因是 action chunking 误差、receding horizon 执行偏差和状态估计误差会在长链中累积。
- 环境泛化失败：ABC 训练到 D 测试存在视觉布局、光照、物体状态和桌面背景差异。未充分增强或未做 failure-aware 后训练的模型容易在 D 环境第一步就偏离。
- 动作精度不足：lift、rotate、stack、place 等任务对末端位置、姿态和 gripper 时序要求高，小 steps 模型常能到达大概位置但无法稳定完成接触/抓取。
- 多峰动作被平均：OFT 或参数 soup 在多种可行动作模式之间平均时，可能生成看似平滑但无效的轨迹。
- normalization/接口问题：早期评估出现 `franka` 与 `new_embodiment` unnorm key 不一致、state 维度不一致、FAST token decode 为 `None` 等问题，这类问题会让模型本身还没被公平评估就失败。

已实施改进：

- 增加服务端/客户端 normalization key 自动解析和维度适配。
- 增加数据 registry 和 CALVIN eval 聚合脚本，避免 data mix 未注册造成隐性失败。
- 增加 top-k checkpoint、summary、磁盘守护和完整 state resume，减少训练中断带来的损坏 checkpoint。
- 增加 `T20/T21/T24/T27/T28` 后训练路径，为 failure replay、anti-forgetting、offline RL、DPO-style ranking 和 checkpoint averaging 留出可执行入口。
- 增加 MoE/adapter/structure policy，使长程任务中不同动作族和失败模式可以走不同 expert 或后训练策略。

## 6. 结论与个人贡献说明

本部分主要贡献是把 StarVLA 从单脚本训练整理成完整工程系统：自由选择 dataset、training policy、base model、action expert 和 structure policy；支持从 base pretrained 从头训练，也支持从已有 checkpoint 后训练；支持 H200 多卡高速训练、磁盘空间守护恢复、tmux 交互式监控、CALVIN websocket 评估、参数 soup ensemble 和公共目录规范。工程上修复了评估 normalization、FAST token decode、GPU 残留进程、batch 自动退化、server/client 并发和 checkpoint 损坏定位等问题。

从技术路线看，世界模型路径有长程规划潜力，但在本次 2 天时限内训练效率和评估稳定性不如 Qwen/MoE VLA 路线；小模型/小 epoch 实验证明 pipeline 可跑通，也暴露了长程泛化和动作精度瓶颈；最优候选模型中 GTY MoE 60k 和 WMH augmented hard v2 在小规模评估中都达到约 12% 的 5/5 成功率，说明 MoE 与数据增强对 CALVIN ABC→D 有明显价值；参数级 heterogeneous ensemble 没有提升，后续应优先做 action-level ensemble、failure-aware post-training 和同构 checkpoint averaging。
