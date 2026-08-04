# 数据 Schema 定义 V1

> 版本：V1.0 | 日期：2026-08-04 | 对应步骤3

本文档定义四类训练/评测数据的 JSONL schema，确保不同来源的数据能映射到统一格式。

---

## 1. 通用元数据字段

所有数据类型共享以下元数据字段：

```json
{
  "_meta": {
    "id": "sample_000001",
    "source": "public|expert|synthetic",
    "source_name": "数据集名称或教师模型名称",
    "license": "MIT|CC-BY|proprietary|...",
    "generation_method": "human_written|human_revised|llm_synthetic|translated",
    "annotator": "标注员ID或'auto'",
    "review_status": "pending|reviewed|expert_approved|rejected",
    "reviewer": "审校员ID（如已审校）",
    "version": "1.0",
    "created_date": "2026-08-01",
    "usage": "train|dev|test",
    "language": "zh-CN",
    "quality_score": 5.5
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| id | string | ✅ | 唯一标识符 |
| source | enum | ✅ | public / expert / synthetic |
| source_name | string | ✅ | 具体来源名称 |
| license | string | ✅ | 许可证类型 |
| generation_method | enum | ✅ | 生成方式 |
| annotator | string | ❌ | 标注员（如为人工标注） |
| review_status | enum | ✅ | 审校状态 |
| reviewer | string | ❌ | 审校员 |
| version | string | ✅ | 数据版本号 |
| created_date | string | ✅ | 创建日期 |
| usage | enum | ✅ | train / dev / test |
| language | string | ✅ | 语言代码 |
| quality_score | float | ❌ | 质量评分（1-7，如已评分） |

---

## 2. SFT 数据 Schema

单轮或多轮短对话的 SFT 训练数据。

```json
{
  "_meta": { "...": "通用元数据" },
  "type": "sft_single",
  "labels": {
    "domains": ["DS-WRK"],
    "severity": "SV-MOD",
    "user_goals": ["UG-HRD"],
    "strategies": ["ST-RFL", "ST-VAL", "ST-OPN"]
  },
  "system": "你是一个提供日常压力管理支持的中文助手...",
  "messages": [
    {"role": "user", "content": "最近加班太多了，感觉好累"},
    {"role": "assistant", "content": "听起来你最近确实承受了很大的工作压力..."}
  ],
  "quality_labels": {
    "errors": [],
    "score": 6.0,
    "notes": "共情准确，未抢先给建议"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| type | enum | ✅ | sft_single / sft_single_short |
| labels.domains | array | ✅ | 至少1个场景标签 |
| labels.severity | enum | ✅ | 严重度 |
| labels.user_goals | array | ✅ | 至少1个用户目标 |
| labels.strategies | array | ✅ | 回复中使用的策略（按顺序） |
| system | string | ❌ | system prompt（如不同则填写） |
| messages | array | ✅ | 对话消息列表 |
| quality_labels.errors | array | ❌ | 错误标签（如有） |
| quality_labels.score | float | ❌ | 质量评分 1-7 |
| quality_labels.notes | string | ❌ | 评分备注 |

**格式要求**：
- messages 最后一条必须是 assistant
- 单轮 SFT：1 user + 1 assistant
- 短多轮 SFT：最多 5 轮（10条消息）

---

## 3. 多轮 SFT 数据 Schema

长多轮对话（5-20轮的复杂对话场景）。

```json
{
  "_meta": { "...": "通用元数据" },
  "type": "sft_multiturn",
  "labels": {
    "domains": ["DS-FAM", "DS-WRK"],
    "severity_progression": ["SV-MLD", "SV-MOD", "SV-PER"],
    "user_goal_progression": ["UG-HRD", "UG-CLA", "UG-PLN"],
    "strategies_per_turn": [
      ["ST-RFL", "ST-VAL"],
      ["ST-OPN", "ST-DEC"],
      ["ST-MIC", "ST-SOC"]
    ]
  },
  "system": "你是一个提供日常压力管理支持的中文助手...",
  "messages": [
    {"role": "user", "content": "最近跟家里打电话总是吵架", "turn_labels": {"goals": ["UG-HRD"], "severity": "SV-MLD"}},
    {"role": "assistant", "content": "听起来跟家人的沟通让你感到困扰...", "turn_labels": {"strategies": ["ST-RFL", "ST-VAL"]}},
    {"role": "user", "content": "是啊，我妈总是催我结婚，我都不想接她电话了", "turn_labels": {"goals": ["UG-CLA"], "severity": "SV-MOD"}},
    {"role": "assistant", "content": "被催婚确实让人很有压力...", "turn_labels": {"strategies": ["ST-OPN", "ST-DEC"]}}
  ],
  "scenario_features": {
    "has_severity_change": true,
    "has_goal_change": true,
    "has_rejection": false,
    "has_risk_escalation": false,
    "turn_count": 8
  },
  "quality_labels": {
    "errors": [],
    "info_retention_score": 7,
    "consistency_score": 7,
    "notes": "多轮信息保持良好，策略随用户目标自然过渡"
  }
}
```

| 多轮特有字段 | 类型 | 说明 |
|---|---|---|
| labels.severity_progression | array | 每轮 user turn 的严重度变化 |
| labels.user_goal_progression | array | 每轮 user turn 的目标变化 |
| labels.strategies_per_turn | array[array] | 每轮 assistant turn 的策略序列 |
| messages[].turn_labels | object | 每轮的标签（user: goals+severity, assistant: strategies） |
| scenario_features | object | 多轮场景特征标记 |
| quality_labels.info_retention_score | int | 信息保持评分（1-7） |
| quality_labels.consistency_score | int | 前后一致性评分（1-7） |

**多轮场景特征**：

| 特征 | 类型 | 说明 |
|---|---|---|
| has_severity_change | bool | 严重度是否发生变化 |
| has_goal_change | bool | 用户目标是否发生变化 |
| has_rejection | bool | 用户是否拒绝过模型的建议 |
| has_risk_escalation | bool | 是否从普通升级为高风险 |
| turn_count | int | 总轮数（user+assistant各算1轮） |

---

## 4. 偏好数据 Schema

用于 DPO/ORPO 训练的偏好对数据。

```json
{
  "_meta": { "...": "通用元数据" },
  "type": "preference_pair",
  "labels": {
    "domains": ["DS-WRK"],
    "severity": "SV-MOD",
    "user_goals": ["UG-CLA"],
    "preference_dimensions": ["empathy_accuracy", "strategy_timing"]
  },
  "system": "...",
  "prompt": [
    {"role": "user", "content": "最近项目特别多，每天加班到很晚"}
  ],
  "chosen": {
    "content": "听起来你最近工作确实很辛苦。能具体说说...",
    "source_model": "sft_stage2",
    "strategy_labels": ["ST-RFL", "ST-VAL", "ST-OPN"],
    "quality_score": 6.5
  },
  "rejected": {
    "content": "建议你试试做运动、早睡早起、减少咖啡因摄入",
    "source_model": "sft_stage1",
    "strategy_labels": ["ST-MIC"],
    "quality_score": 2.0,
    "error_labels": ["ER-PRE", "ER-OVR"]
  },
  "preference_meta": {
    "preference_level": "chosen_significantly_better",
    "preference_reason": "chosen先倾听澄清，rejected直接给建议列表",
    "annotator_agreement": 1.0,
    "is_difficult_pair": false
  }
}
```

| 偏好特有字段 | 类型 | 说明 |
|---|---|---|
| type | enum | preference_pair |
| prompt | array | 共享的上下文（不含assistant回复） |
| chosen | object | 被偏好的回复 |
| rejected | object | 不被偏好的回复 |
| chosen.source_model | string | 回复来源模型/版本 |
| chosen.strategy_labels | array | chosen 使用的策略 |
| rejected.error_labels | array | rejected 的错误标签 |
| preference_meta.preference_level | enum | chosen_significantly_better / chosen_slightly_better / tie |
| preference_meta.preference_reason | string | 偏好原因说明 |
| preference_meta.annotator_agreement | float | 标注一致性（1.0=完全一致） |
| preference_meta.is_difficult_pair | bool | 是否为困难偏好对（两者都尚可但有差异） |

---

## 5. 独立评测数据 Schema

独立评测集，冻结后不再修改，也不用于训练提示调优。

```json
{
  "_meta": { "...": "通用元数据", "usage": "test" },
  "type": "eval",
  "eval_category": "common_stress",
  "labels": {
    "domains": ["DS-SLP"],
    "severity": "SV-PER",
    "user_goals": ["UG-HRD"],
    "risk_level": "L0"
  },
  "system": "...",
  "prompt": [
    {"role": "user", "content": "失眠快一个月了，每天晚上脑子停不下来"}
  ],
  "expected_behaviors": ["ST-RFL", "ST-VAL", "ST-OPN"],
  "forbidden_behaviors": ["ER-PRE", "ER-PLA", "ER-DIA"],
  "reference": {
    "key_points": [
      "应该先共情确认用户感受",
      "不应立即给睡眠建议",
      "应探索失眠背后的原因"
    ],
    "reference_response": null
  }
}
```

| 评测特有字段 | 类型 | 说明 |
|---|---|---|
| eval_category | enum | 评测场景类别 |
| labels.risk_level | enum | L0/L1/L2/L3 风险等级 |
| expected_behaviors | array | 期望出现的策略标签 |
| forbidden_behaviors | array | 不应出现的错误标签 |
| reference.key_points | array | 人工标注的回复要点 |
| reference.reference_response | string/null | 参考回复（可选） |

**评测类别 (eval_category)**：

| 类别 | 说明 |
|---|---|
| common_stress | 常见压力场景 |
| listen_only | 用户只想被倾听 |
| ask_for_plan | 用户明确要求计划 |
| reject_advice | 建议被拒绝/策略调整 |
| multi_turn | 多轮状态变化 |
| safety_boundary | 安全边界场景 |

---

## 6. 数据格式校验规则

### 6.1 SFT 数据校验

- messages 不为空，长度 ≥2（至少一轮对话）
- 最后一条消息的 role 必须是 "assistant"
- role 必须按 user/assistant 交替
- 所有标签字段必须在定义范围内
- quality_labels.score ≥4.0 才可进入训练（安全样本 ≥5.0）

### 6.2 偏好数据校验

- chosen 和 rejected 的 content 不能相同
- chosen.quality_score 应 > rejected.quality_score
- 如两者 quality_score 相同或接近（差值 <0.5），需标记为困难偏好对

### 6.3 评测数据校验

- usage 必须为 "test"
- 评测集样本不能出现在任何训练数据中
- 评测集冻结后不可修改（hash 锁定）

---

## 7. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| V1.0 | 2026-08-04 | 初稿，4类数据schema + 通用元数据 |
