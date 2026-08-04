# 数据来源与版本字段规范

> 版本：V1.0 | 日期：2026-08-04 | 对应步骤3

---

## 1. 数据来源分类

### 1.1 三大来源

| 来源编码 | 名称 | 说明 | SFT目标占比 |
|---|---|---|---|
| public | 公开/授权数据 | 开源数据集、授权使用的外部数据 | 25%–40% |
| expert | 专家编写/审校数据 | 领域专家或受训标注员编写和审校 | 20%–30% |
| synthetic | LLM合成数据 | 教师模型生成，补充长尾覆盖 | 30%–50% |

### 1.2 公开数据许可审计

每个公开数据集入库前必须填写：

```json
{
  "dataset_name": "ESConv",
  "source_url": "https://github.com/thu-coai/ESConv",
  "license": "Apache-2.0",
  "license_verified": true,
  "usage_allowed": ["research", "training"],
  "modification_allowed": true,
  "redistribution_allowed": true,
  "contains_pii": false,
  "contains_clinical_content": false,
  "language": "zh-CN",
  "original_format": "json",
  "conversion_script": "data_processing/convert_esconv.py",
  "conversion_date": "2026-08-04",
  "notes": "情感支持对话数据集，已过滤含诊断性内容样本"
}
```

### 1.3 不可使用的数据（红线）

| 红线 | 说明 |
|---|---|
| 许可不明 | 无法确认许可证的数据 → 不入库 |
| 禁止商用 | 如许可证仅限 non-commercial，本项目的许可要求可能冲突 |
| 包含真实PII | 未脱敏的真实个人信息 → 不入库 |
| 临床诊断内容 | 包含明确诊断、药物推荐、治疗方案的内容 → 不入库 |
| 未授权真实对话 | 未获得明确授权的真实用户对话 → 不入库 |

---

## 2. 数据版本命名

### 2.1 格式

```
{project}-{data_type}-{size}-{date}-{version}
```

### 2.2 示例

```
sm-sft-30k-20260804-v1
sm-pref-10k-20260804-v1
sm-eval-3k-20260804-v1
sm-multiturn-5k-20260804-v1
sm-safety-3k-20260804-v1
```

### 2.3 版本递增规则

| 变更类型 | 版本号变化 | 示例 |
|---|---|---|
| 新增样本 | v1 → v2 | 从20k扩充到30k |
| 修正标签 | v1 → v1.1 | 修复错误标签 |
| 重新切分 | v1 → v1.2 | 重新split train/dev/test |
| 重大变更 | v1 → v2.0 | 更换标注指南或schema |

---

## 3. 数据处理链路记录

每条数据必须记录完整的处理链路：

```json
{
  "provenance": {
    "original_source": {
      "type": "public|expert|synthetic",
      "name": "来源名称",
      "raw_id": "原始数据ID",
      "raw_format": "原始格式"
    },
    "processing": [
      {"step": "format_conversion", "script": "convert_esconv.py", "date": "2026-08-01"},
      {"step": "language_check", "script": "check_lang.py", "date": "2026-08-01"},
      {"step": "pii_scan", "script": "scan_pii.py", "date": "2026-08-02"},
      {"step": "dedup", "method": "exact+minhash", "date": "2026-08-02"},
      {"step": "labeling", "annotator": "A001", "date": "2026-08-03"},
      {"step": "review", "reviewer": "E001", "date": "2026-08-04", "status": "approved"}
    ],
    "final_version": "sm-sft-30k-20260804-v1",
    "final_usage": "train"
  }
}
```

---

## 4. 数据切分规范

### 4.1 切分原则

| 原则 | 说明 |
|---|---|
| 按来源分组切分 | 同一原始数据集的样本不跨 train/dev/test |
| 按对话主体分组 | 同一用户/人物的多轮对话不跨切分 |
| 按合成模板分组 | 同一模板生成的变体不跨切分 |
| 按语义近邻分组 | embedding相近的样本不跨切分 |
| Stratified split | 按场景类型和严重度的分布分层采样 |

### 4.2 目标分布

| 数据池 | train | dev | test |
|---|---|---|---|
| SFT数据 | 85% | 10% | 5% |
| 偏好数据 | 85% | 10% | 5% |
| 安全数据 | 80% | 10% | 10% |
| 独立评测集 | — | — | 100%（完全独立） |

### 4.3 泄漏审计

| 检查项 | 方法 | 阈值 |
|---|---|---|
| 精确重复 | 文本hash相同 | 0容忍 |
| 近重复 | MinHash Jaccard ≥0.8 | 0容忍（train↔test之间） |
| 语义近邻 | embedding cosine ≥0.95 | 0容忍（train↔test之间） |
| 同源模板 | 相同生成模板ID | 0容忍（train↔test之间） |

---

## 5. 数据使用范围

### 5.1 使用范围定义

| 范围 | 编码 | 说明 |
|---|---|---|
| 无限制 | unrestricted | 可用于训练、评测、公开发布 |
| 训练+评测 | train_eval | 可用于训练和内部评测，不公开发布 |
| 仅评测 | eval_only | 仅用于评测，不进入训练集 |
| 仅内部 | internal_only | 仅内部研究使用 |

### 5.2 Data Card 模板

每个数据版本发布时生成 Data Card：

```markdown
# Data Card: sm-sft-30k-20260804-v1

## 基本信息
- 名称: sm-sft-30k-20260804-v1
- 版本: v1
- 创建日期: 2026-08-04
- 样本数: 30,000
- 语言: zh-CN

## 来源构成
| 来源 | 样本数 | 占比 |
|---|---|---|
| 公开数据 | 10,000 | 33% |
| 专家数据 | 7,000 | 23% |
| LLM合成 | 13,000 | 43% |

## 标签分布
| 场景 | 占比 |
|---|---|
| DS-WRK | 22% |
| DS-LRN | 18% |
| DS-REL | 15% |
| ... | ... |

## 严重度分布
| 等级 | 占比 |
|---|---|
| SV-MLD | 30% |
| SV-MOD | 40% |
| SV-PER | 20% |
| SV-IMP | 7% |
| SV-RSK | 3% |

## 使用限制
- 范围: train_eval
- 不可再分发: false
- 伦理审查: 已通过

## 处理记录
- PII扫描: 通过（0条检出）
- 去重: 精确去重+MinHash
- 泄漏审计: train/test 近重复率 0%
```

---

## 6. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| V1.0 | 2026-08-04 | 初稿 |
