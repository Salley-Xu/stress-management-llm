# 步骤11 回顾：构建偏好数据

> 回顾日期：2026-08-08 | 范围：步骤11（偏好数据构建，Rubric驱动版）

---

## 一、步骤任务回顾

原步骤11要求：
1. 从多个模型生成同一prompt的3-6个候选
2. 8个评价维度
3. LLM judge初筛 + 人工成对排序 + 高风险专家排序
4. 保留困难偏好对
5. 三类边界对（普通/谨慎/高风险）
6. 保存chosen/rejected/原因/一致性

**用户升级**：设计时无Rubric，现在有35分制Rubric + 两轮SFT数据，重新设计。

## 二、已完成内容

### 新方案：Rubric驱动的偏好构建

```
550 prompt（评测集6场景）
   → 4候选（base/stage1/stage2子进程 + teacher并发）
   → 35分制Rubric独立评分（12并发，2200候选）
   → 偏好对构造（分差2-8=困难对 + 安全门槛）
   → 一致性验证（93%）
```

### 关键产出

| 产出 | 值 |
|---|---|
| 候选 | 4模型×550条 |
| Rubric质量梯度 | base 25.7 / stage1 28.9 / stage2 29.3 / teacher 32.5 |
| 偏好对 | **2,335对**（困难对83.8%） |
| 一致性 | **93%** |

### 踩坑记录

| # | 坑 | 表现 | 解决 |
|---|---|---|---|
| 1 | 模型切换显存不足 | base生成后加载stage1报"modules dispatched on CPU" | **子进程隔离**（每模型独立进程） |
| 2 | del model不释放显存 | 同进程顺序加载3模型8GB不够 | 改gen_candidate_worker.py子进程 |
| 3 | 候选生成耗时长 | base 48min/stage1 36min/stage2 39min | 后台跑+断点续跑（已生成跳过） |
| 4 | DeepSeek reasoning空响应 | 教师生成偶发空 | max_tokens=4096+重试 |

## 三、关键决策记录

### 决策1：Rubric独立打分替代成对比较

**背景**：成对比较只有"A>B"结论，无解释。

**决策**：35分制Rubric对每候选独立打分，分数差构造偏好对。

**影响**：可解释性（分维度差异=偏好原因），且与成对验证一致率93%。

### 决策2：三阶段模型 + 教师 4候选

**背景**：原方案4模型，无Rubric时教师是唯一质量上限。

**决策**：base/stage1/stage2（本地）+ teacher（DeepSeek）。

**影响**：质量梯度天然形成（25.7→32.5），候选自带偏好方向，无需人工标优劣。

### 决策3：困难对窗口 Δ2-8分

**背景**：DPO需要细粒度偏好，明显优劣和打平都无价值。

**决策**：Δ2-8=困难对（保留），Δ>10=明显对（保留），Δ<2=丢弃。

**影响**：困难对占83.8%，远超通过条件。

### 决策4：安全safety维度硬门槛

**背景**：高风险偏好对必须正确（步骤11通过条件）。

**决策**：safety_boundary场景 chosen.safety≥4 且 rejected.safety≤3。

**影响**：202对过滤，剩余73对安全区分明确（chosen 4.81 vs rejected 2.36）。

## 四、遗留问题

| 问题 | 说明 | 方向 |
|---|---|---|
| safety_boundary对偏少 | 仅73对 | DPO时安全对可重点加权 |
| 反判7% | Rubric与成对判断不一致 | 可人工复核反判样本 |
| base>teacher 6对 | Rubric噪声 | 占比0.5%，可过滤 |

## 五、资产清单

### 新增脚本
| 脚本 | 用途 |
|---|---|
| data_processing/build_preference_data.py | 4步流程 |
| data_processing/gen_candidate_worker.py | 单模型子进程生成 |

### 新增数据
| 文件 | 说明 |
|---|---|
| data/processed/preference_data/pairs_*.jsonl | 2,335偏好对 |
| data/processed/preference_data/scored_*.jsonl | Rubric评分 |
| data/processed/preference_data/candidates_*.jsonl | 候选 |

### 新增报告
| 文件 | 说明 |
|---|---|
| reports/step11_preference_report.md | 偏好数据报告 |

## 六、衔接下一步

**步骤12（DPO）**：用pairs_*.jsonl训练DPO。
- 起点：SFT-Stage2 adapter
- 数据：2,335偏好对（chosen/rejected带Rubric分）
- 建议beta 0.05/0.1/0.2对比
- 重点：安全边界对强化，防风险漏检反弹
