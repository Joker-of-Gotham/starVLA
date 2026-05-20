# WMH StarVLA CALVIN Branch

这是 WMH 在 StarVLA 上针对 CALVIN ABC -> D 的实验分支。训练只使用 CALVIN ABC，CALVIN D 只用于 closed-loop evaluation；不使用任何上游 action-trained checkpoints。

## 重点阅读

完整路线选择、failure pattern、各分支结果和后续实验优先级见：

**[CALVIN ABC -> D 技术路线选择论证与 Failure Pattern 分析报告](examples/calvin_autoresearch/docs/calvin_abc_d_技术路线与failure分析报告_中文整理版.md)**

详细文档、脚本说明、mirror augmentation 分析、checkpoint 下载和 public 测评说明见：

**[WMH CALVIN AutoResearch 详细 README](examples/calvin_autoresearch/README.md)**

在线查看时请切到 GitHub 的 `WMH` 分支：

```text
https://github.com/Joker-of-Gotham/starVLA/tree/WMH
```

## 总览

当前 WMH 分支的主要路线是 failure-driven data/model adaptation，而不是单纯延长 baseline 训练。

核心改动：

- CALVIN ABC 专用 dataloader、data registry 和训练配置。
- hard-task balanced sampling，重点覆盖 slider、drawer、light/LED、directional push 等 D-critical tasks。
- controlled language paraphrase 和 task-aware image augmentation。
- left/right mirror augmentation 及其可视化诊断。
- QwenGR00T / connector / state / LoRA / MoE95k continuation 支持。
- 多 GPU CALVIN D evaluation、public-only evaluation runner 和详细 metrics 汇总。
- `./wmh` 一键入口，用于训练、评测、tail log、status 和 finalize。

当前 verified 主路线：

```text
hard-task balanced ABC training
+ controlled language paraphrase
+ task-aware image augmentation
```

主要结果：

| Branch | N | Avg Seq Len | SR@1 | SR@5 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `base8k` | 300 | 1.050 | 54.0% | 3.7% | reference baseline |
| `lora2000` | 300 | 1.630 | 64.0% | 9.7% | LoRA 有效 |
| `aug_hardv2` | 300 | 1.847 | 72.0% | 12.0% | 当前 WMH verified best |
| `mirror_hardv2` | 300 | 1.753 | 72.7% | 9.7% | mirror 有效，但作为 diagnostic branch |
| `no_mirror_MoE95k_LoRA` | 100 | 1.940 | 77.0% | 9.0% | HF 已公开，需补 n300/n1000 |

## Public 快速测评

如果你只能访问公共区 `/public/seven`，可直接使用 WMH 公共 checkpoint 区：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/checkpoints
```

当前默认 checkpoint：

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/checkpoints/latest_latest_pytorch_model.pt
```

快速评测示例：

```bash
export PUBLIC=/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin
source ${PUBLIC}/shared/runtime/starvla_env.sh

export STARVLA_ROOT=${PUBLIC}/members/WMH/code/latest_starVLA_moe_lora
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH:-}
cd ${STARVLA_ROOT}

MEMBER=YOUR_NAME
CKPT=${PUBLIC}/members/WMH/checkpoints/latest_latest_pytorch_model.pt
RUN_ID=eval_wmh_public_d_n300_$(date +%m%d_%H%M%S)
OUT=${PUBLIC}/members/${MEMBER}/reports/${RUN_ID}
LOG=${PUBLIC}/members/${MEMBER}/logs/${RUN_ID}.log
mkdir -p "$(dirname "${LOG}")" "${OUT}"

CKPT=${CKPT} \
EVAL_LOG_DIR=${OUT} \
TOTAL_SEQUENCES=300 \
GPU_IDS=0,1,2,3 \
WORKERS_PER_GPU=1 \
BASE_PORT=7400 \
CALVIN_SEND_STATE=1 \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh > "${LOG}" 2>&1

${STARVLA_PYTHON} examples/calvin_autoresearch/scripts/summarize_eval_metrics.py "${OUT}/metrics.json"
```

把 `MEMBER=YOUR_NAME` 改成自己的名字，评测结果会写到自己的 public member 目录。

更多 checkpoint、Hugging Face 下载、mirror 增强和 failure analysis 见：

```text
examples/calvin_autoresearch/README.md
```

## Hugging Face 权重

no-mirror MoE95k + LoRA 权重已上传到 Hugging Face：

```bash
git lfs install
git clone https://huggingface.co/wwwafwet/no-mirror_MoE95k_LoRA
```

推荐权重：

```text
checkpoints/steps_7000_pytorch_model.pt
```

## 上游 StarVLA

本分支基于 StarVLA 开发。上游项目文档请参考原仓库和 `docs/` 目录；WMH CALVIN 实验入口以本 README 和 `examples/calvin_autoresearch/README.md` 为准。
