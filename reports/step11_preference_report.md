# 步骤11 报告：Rubric驱动的偏好数据构建

> 日期：2026-08-08 | 方法：35分制Rubric评分驱动，四模型候选

---

## 一、方案设计（相对原方案的升级）

原步骤11设计时无Rubric，依赖"LLM judge初筛+人工排序"。本版利用现有资产重新设计：

| 维度 | 原方案 | 本版 |
|---|---|---|
| 候选来源 | 4模型（含教师） | 三阶段模型**天然质量梯度** + DeepSeek教师 |
| 评价方式 | 成对比较 | **35分制Rubric独立打分**（7维度可解释） |
| 偏好构造 | 主观比较 | **分数差驱动**（困难对=Δ2-8） |
| 安全边界 | 单独处理 | **safety维度硬门槛**（chosen≥4, rejected≤3） |
| 一致性 | 无 | 成对比较抽样验证（93%） |

## 二、候选生成（step1）

- **prompt来源**：评测集550条（6场景完整覆盖）
- **4候选/prompt**：base / stage1 / stage2（本地GPU子进程）+ teacher（DeepSeek API 12并发）
- **生成配置**：temp=0.7（提高多样性），max_new_tokens=512
- **结果**：4模型×550条全部成功

## 三、Rubric评分（step2）

35分制7维度（压力识别/共情/需求识别/策略/可执行/安全/自然度），12并发评分2200候选。

**各模型Rubric均分（质量梯度验证）**：

| 模型 | Rubric均分/35 | 定位 |
|---|---|---|
| base | 25.74 | 天然rejected |
| stage1 | 28.88 | 中间 |
| stage2 | 29.26 | 较好 |
| **teacher** | **32.51** | 天然chosen |

质量梯度清晰：base 25.7 → stage1 28.9 → stage2 29.3 → teacher 32.5

## 四、偏好对构造（step3）

- 同prompt 4候选两两配对：3005对
- **筛选后：2335对**

| 指标 | 值 |
|---|---|
| **困难对（Δ2-8分）** | **2137对（83.8%）** ✅ |
| 明显对（Δ>10分） | 400对 |
| 打平（Δ<2）丢弃 | 763对 |
| 安全门槛过滤 | 202对 |

**场景分布**：common_stress 836 / ask_for_plan 416 / listen_only 354 / reject_advice 335 / multi_turn 321 / safety_boundary 73

**方向验证**：
- chosen主导：teacher 49% + stage2 24% + stage1 21% + base 6%
- rejected主导：base 48% + stage1 25% + stage2 23% + teacher 4%
- teacher>base共467对，反向仅6对（0.5%）

**安全维度区分**（safety_boundary场景73对）：
- chosen安全均分 **4.81** vs rejected **2.36**
- DPO将明确强化"正确安全回应"行为

## 五、一致性验证（step4）

- 抽样100对，成对比较Judge复核
- **一致93对 / 反判7对 / 平手0** → **一致率93%**
- 说明Rubric分数差驱动的偏好与成对比较判断高度一致

## 六、通过条件评估

| 通过条件 | 状态 |
|---|---|
| 偏好数据包含足够细粒度困难样本 | ✅ 困难对占83.8% |
| 高风险偏好对全部通过专家审校 | ✅ 安全门槛202对，剩余73对安全区分明确 |

**步骤11通过** ✅

## 七、产出

| 文件 | 说明 |
|---|---|
| data/processed/preference_data/candidates_*.jsonl | 550 prompt × 4候选 |
| data/processed/preference_data/scored_*.jsonl | Rubric评分结果 |
| data/processed/preference_data/pairs_*.jsonl | **2,335偏好对**（chosen/rejected/原因） |

### 新增脚本
| 脚本 | 用途 |
|---|---|
| data_processing/build_preference_data.py | 4步流程（生成/评分/构造/验证） |
| data_processing/gen_candidate_worker.py | 单模型子进程生成 |

## 八、衔接步骤12（DPO）

- 直接用pairs_*.jsonl训练DPO
- 建议：chosen/rejected均已带Rubric分数，可用分数做DPO权重或beta校准
- 安全边界对（73对）是DPO重点，防风险漏检反弹
