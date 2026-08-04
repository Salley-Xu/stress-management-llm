# 步骤3总结：设计数据体系与统一schema

> 完成日期：2026-08-04

---

## 一、步骤任务回顾

根据纯模型训练版，步骤3包含7项任务：

1. 建立压力场景标签（10类）
2. 建立用户目标标签（6类）
3. 建立压力程度标签（5级）
4. 建立支持策略标签（11类）
5. 建立质量与错误标签（9类）
6. 定义四类数据 schema（SFT/多轮/偏好/评测）
7. 为每条数据保留来源、许可、生成方式、标注者、审校状态、版本和可用范围

---

## 二、已完成内容

### 2.1 产出文档

| 文件 | 内容 | 篇幅 |
|---|---|---|
| [docs/data_taxonomy_v1.md](../docs/data_taxonomy_v1.md) | 数据标签体系 V1 | ~210行 |
| [docs/data_schema_v1.md](../docs/data_schema_v1.md) | JSONL Schema 定义 | ~280行 |
| [docs/annotation_guide_v1.md](../docs/annotation_guide_v1.md) | 标注指南 V1 | ~260行 |
| [docs/data_provenance_spec.md](../docs/data_provenance_spec.md) | 数据来源与版本规范 | ~220行 |

### 2.2 标签体系摘要

| 标签类型 | 数量 | 示例 |
|---|---|---|
| 压力场景 (DS-*) | 10 | DS-LRN, DS-WRK, DS-CAR, DS-INT, DS-REL, DS-FAM, DS-FIN, DS-SLP, DS-MIG, DS-PRC |
| 用户目标 (UG-*) | 6 | UG-HRD, UG-CLA, UG-STB, UG-DEC, UG-PLN, UG-REV |
| 压力程度 (SV-*) | 5 | SV-MLD, SV-MOD, SV-PER, SV-IMP, SV-RSK |
| 支持策略 (ST-*) | 11 | 倾听(4) + 认知(2) + 行动(2) + 稳定资源(3) |
| 质量错误 (ER-*) | 9 | ER-PRE, ER-PLA, ER-OVR, ER-IMP, ER-REP, ER-FGT, ER-DIA, ER-OVRF, ER-DEP |

### 2.3 Schema 定义

| Schema | 用途 | 关键字段 |
|---|---|---|
| sft_single | 单轮/短多轮SFT | messages + labels + quality_labels |
| sft_multiturn | 长多轮SFT | 多轮专属：severity_progression, goal_progression, scenario_features |
| preference_pair | DPO/ORPO偏好对 | chosen/rejected + preference_meta |
| eval | 独立评测集 | expected_behaviors + forbidden_behaviors + reference |

### 2.4 标注指南关键规则

- **就高不就低**：严重度判定取最高风险信号
- **多标签允许**：场景、目标、策略均可多标签
- **质量门槛**：SFT样本 score ≥4.0（安全 ≥5.0）
- **一致性目标**：场景 κ≥0.75, 严重度 κ≥0.85
- **SV-RSK强制双人标注** + 专家复核

### 2.5 数据来源规范

- **三来源混合**：public 25-40% / expert 20-30% / synthetic 30-50%
- **5条红线**：许可不明、禁止商用、含PII、临床内容、未授权对话 → 不入库
- **处理链路**：每样本记录完整 provenance（原始来源 → 每步处理 → 最终版本）
- **泄漏审计**：train↔test 精确重复0容忍，近重复0容忍

---

## 三、与旧版标签的兼容

原有 [scenario_task_taxonomy.md](../docs/scenario_task_taxonomy.md)（14领域/6严重度/8目标/22策略）作为背景参考保留。新版标签体系在旧版基础上进行了以下调整：

| 变更 | 说明 |
|---|---|
| 领域精简 | 14→10，合并"学习"+"考试"，新增"长期任务和拖延" |
| 严重度合并 | 6级→5级，S-RISK+S-EMER→SV-RSK |
| 目标精简 | 8→6，移除 G-RES 和 G-COM |
| 策略精简 | 22→11，按功能类别合并 |
| 错误标签 | 11→9，合并相关类别 |

---

## 四、踩坑记录

无重大踩坑。本步骤为纯文档设计工作，前面标签体系已有基础，主要工作是对齐和收敛。

---

## 五、下一步

进入**步骤4：收集公开或授权数据**。

前置条件：需要网络访问 HuggingFace / GitHub 下载公开数据集。
