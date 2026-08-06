# 训练数据质量评估报告

> 日期：2026-08-06 | 评估方法：DeepSeek v4-flash 按100分制评分标准

---

## 一、评估结论

**SmileChat 数据大量不合格（合格率仅8.5%），已按混合策略处理。**

## 二、评估方法

- 评估标准：[training_data_quality_standard.md](../docs/training_data_quality_standard.md)（100分制7维度）
- 抽样：每来源100条，共300条
- 评估模型：DeepSeek v4-flash（temperature 0.3，稳定评分）
- 成功率：296/300（98.7%）

## 三、质量分布结果

### 总体

| 指标 | 值 |
|---|---|
| 平均总分 | 81.6/100 |
| 平均安全分 | 14.6/15 |
| 分类 | core 145 / normal 58 / low_weight 11 / dpo_rejected 24 / delete 58 |

### 各来源对比

| 来源 | 数量 | core | normal | low | dpo | delete | **合格率(≥70)** |
|---|---|---|---|---|---|---|---|
| **专家数据** | 491 | 88 | 9 | 2 | 0 | 0 | **99%** ✅ |
| **DeepSeek合成** | 5,377 | 56 | 42 | 0 | 0 | 0 | **98%** ✅ |
| **SmileChat** | 37,300 | 1 | 7 | 9 | 24 | 58 | **8.5%** ❌ |

### SmileChat 问题分析

- 回复质量分极低（3-7/20分）
- 安全分尚可（10-15/15分）
- 主要问题：counselor回复模板化、说教、理解浅

## 四、处理方案（混合策略）

| 来源 | 处理 | 保留量 |
|---|---|---|
| 专家数据 | 全保留（99%合格） | 491 |
| DeepSeek合成 | 全保留（98%合格） | 5,377 |
| SmileChat | 随机抽样3,000作补充 | 3,000 |
| **最终数据池** | | **8,868** |

### 最终切分

| Split | 数量 | 来源构成 |
|---|---|---|
| train | 7,531 | 合成4,568 / SmileChat 2,555 / 专家408 |
| dev | 882 | 合成520 / SmileChat 313 / 专家49 |
| test | 455 | 合成289 / SmileChat 132 / 专家34 |
| 泄漏 | 0 | |

## 五、质量策略总结

1. **质量优先**：核心训练数据为专家(99%) + 合成(98%)，共5,868条高质量样本
2. **SmileChat降级**：从37K降为3K补充，保留数据多样性但控制噪声
3. **可复现**：评估脚本 evaluate_training_data.py，结果保存在 data/processed/eval_reports/
4. **后续**：如需更多数据，优先扩充专家/合成，而非SmileChat

## 六、脚本与产物

| 文件 | 说明 |
|---|---|
| [data_processing/evaluate_training_data.py](../data_processing/evaluate_training_data.py) | 100分制抽样评估 |
| [data_processing/build_final_pool.py](../data_processing/build_final_pool.py) | 混合策略重建数据池 |
| [docs/training_data_quality_standard.md](../docs/training_data_quality_standard.md) | 100分制评分标准 |
| data/processed/final_pool_v1.jsonl | 最终数据池（8,868条） |
| data/processed/final_split/{train,dev,test}.jsonl | 最终切分 |
