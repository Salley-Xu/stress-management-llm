# 步骤5总结：构建专家编写与专家审校数据

> 完成日期：2026-08-04

---

## 一、步骤任务回顾

根据纯模型训练版，步骤5包含7项任务：

1. 邀请领域专家参与数据编写和审校
2. 编写覆盖难点的高质量样本（7类难点）
3. 对公开数据和LLM合成数据进行专家修订
4. 对回复按统一量表评分（7维度）
5. 高风险样本双人审校和仲裁
6. 建立专家金标准小集
7. 真实对话数据授权、脱敏和伦理审查

建议占比：最终 SFT 数据的 20%–30%

---

## 二、已完成内容

### 2.1 产出文档

| 文件 | 内容 | 篇幅 |
|---|---|---|
| [docs/expert_data_design.md](../docs/expert_data_design.md) | 专家数据设计：7类难点覆盖矩阵、编写模板、写作原则 | ~220行 |
| [docs/expert_review_guidelines.md](../docs/expert_review_guidelines.md) | 专家审校指南：审校流程、评分表、高风险特殊流程、一致性计算 | ~240行 |
| [docs/gold_standard_set_spec.md](../docs/gold_standard_set_spec.md) | 金标准集规范：准入条件、覆盖矩阵、LLM Judge校准流程 | ~250行 |

### 2.2 设计要点

**7类难点覆盖** — 专家数据必须主动设计以下场景：

| 编号 | 难点 | 目标数量 |
|---|---|---|
| E1 | 用户只想被倾听 | 200-300 |
| E2 | 多压力源交织 | 200-300 |
| E3 | 用户信息不完整 | 150-200 |
| E4 | 建议被拒绝 | 150-200 |
| E5 | 已尝试方法无效 | 150-200 |
| E6 | 目标多轮变化 | 150-200 |
| E7 | 安全边界/高风险 | 200-300 |

**专家资质分层**：
- 编写员：心理学本科/研究生
- 审校员：2年+咨询经验
- 仲裁专家：5年+临床经验
- 替代方案：培训资深标注员 + 专家仅审校高风险样本

**金标准集**：
- MVP 50条 → 正式版 100-150条
- 用作 LLM Judge 校准（每月 Pearson r ≥0.80）
- 不可用于训练，仅评测和校准
- 每6个月审查防污染

**高风险专项**：
- 安全边界权重×2
- 安全边界 ≤3分 → 直接拒绝入库
- 100% 双人审校 + 专家终审

---

## 三、与现有文档的关联

| 本文档 | 关联文档 |
|---|---|
| expert_data_design.md | → data_taxonomy_v1.md (标签), annotation_guide_v1.md (标注标准) |
| expert_review_guidelines.md | → training_eval_rubric.md (评分维度), behavior_spec.md (行为目标) |
| gold_standard_set_spec.md | → data_schema_v1.md (评测schema), safety_spec.md (安全标准) |

---

## 四、实际执行（2026-08-05/06）

### 4.1 用户设计Rubric规范

用户提供了两套评分规范，均已规范化到项目文档：
- **35分制专家Rubric**（docs/expert_rubric.md）：7维度×0-5分，用于专家数据审校
- **100分制质量标准**（docs/training_data_quality_standard.md）：用于训练数据筛选

### 4.2 DeepSeek生成专家数据

用 DeepSeek v4-flash 按专家角色生成，覆盖7类难点场景：

| 批次 | 生成时间 | core(≥28分) |
|---|---|---|
| 第一批 | 17:12 | 101 |
| 第二批 | 19:16 | 193 |
| 第三批 | 22:33 | 185 |
| **合计** | | **479 core + 24 normal** |

**平均Rubric评分 33.3/35**，核心占比95%。

### 4.3 标签规范化

专家数据使用Rubric风格标签（stress_type/intent/strategy），通过 [normalize_expert_labels.py](../data_processing/normalize_expert_labels.py) 映射到项目标准（DS-*/UG-*/ST-*）。

### 4.4 踩坑记录

| 坑 | 解决 |
|---|---|
| DeepSeek json_object 空响应 | 移除response_format |
| 多轮JSON截断（1024 tokens不足） | 提高到2048 |
| 偶发空响应 | 加重试逻辑 |
| 策略字段乱码（中文描述） | 严格映射过滤 |

---

## 五、下一步

进入**步骤6：使用LLM扩展合成数据**。
