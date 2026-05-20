# SII SummerCamp StarVLA 报告索引

本文件只作为报告入口。技术文档和测评报告已经分开维护，避免技术路线、失败分析和结果表混在一起。

## 1. 技术报告

见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。

内容包括：

- 技术路线选择论证。
- base model、action expert、world model、MoE 的选择逻辑。
- 训练策略设计和数学目标。
- 后训练、failure replay、offline RL、DPO、distillation 的理论逻辑。
- Failure Pattern 分类、根因分析、典型失败案例和改进方案。
- 基于现有 CALVIN 结果的进一步技术判断。

## 2. CALVIN ABC→D 测评报告

见 [CALVIN_EVALUATION_REPORT.md](CALVIN_EVALUATION_REPORT.md)。

内容包括：

- CALVIN ABC→D 测评协议。
- Task 1 到 Task 5 成功率定义。
- Average Chain Length 计算方式。
- 当前可解析的完整测评结果表。
- 最优模型、世界模型、小模型、ensemble 的结果对比与分析。

## 3. 工程使用文档

见 [SUMMERCAMP_STARVLA.md](SUMMERCAMP_STARVLA.md)。

内容包括：

- 离线环境配置。
- 数据、模型、checkpoint 怎么放。
- 交互式训练、续训、后训练、评估、ensemble 怎么操作。
- 主菜单每个选项的含义和选择建议。

## 4. Policy 矩阵

见 [POLICY_MATRIX.md](POLICY_MATRIX.md)。

内容包括：

- Training Policy `T01-T28` 完整表。
- Structure Policy `S01-S32` 完整表。
- 从头训练、后训练、OFT、FAST、PI/GR00T、世界模型路线的组合规则。
