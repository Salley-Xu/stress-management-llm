# 步骤8：训练前基线报告

> 日期：2026-08-06 | 模型：Qwen2.5-7B-Instruct（零样本）

---

## 一、基线评测设置

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-7B-Instruct（4-bit量化） |
| 评测集 | eval_set_v1.jsonl（550条，6场景） |
| 解码 | temperature=0.1, do_sample=False |
| 生成 | max_tokens=512 |
| 成功率 | 550/550（0失败） |
| 平均回复 | 221.7字符 |
| 生成速度 | 5.39s/条 |

## 二、场景分布

| 场景 | 数量 | 平均长度 |
|---|---|---|
| common_stress | 186 | 242 |
| ask_for_plan | 84 | 223 |
| listen_only | 87 | 95 |
| reject_advice | 68 | 373 |
| multi_turn | 67 | 179 |
| safety_boundary | 58 | 217 |

## 三、错误类型分布

### LLM评估（DeepSeek，550条全量）

> 每条样本可含多类错误。平均质量分 3.60/5。

| 错误类型 | 数量 | 占比 |
|---|---|---|
| **ERR-PREADVICE 抢先建议** | 111 | **20.2%** |
| **ERR-OVERLOAD 建议过载** | 86 | **15.6%** |
| **ERR-STRATEGY 策略时机不当** | 75 | **13.6%** |
| **ERR-EMPATHY 共情空泛** | 64 | **11.6%** |
| ERR-RISKMISS 风险漏检 | 23 | 4.2% |
| ERR-EMOTION 情绪识别不准确 | 17 | 3.1% |
| ERR-CONTEXT 多轮状态丢失 | 8 | 1.5% |
| ERR-UNDERSTAND 压力源理解错误 | 6 | 1.1% |
| ERR-ACTIONABLE 建议不可执行 | 5 | 0.9% |
| ERR-NORMALIZE 错误正常化 | 4 | 0.7% |
| ERR-REPEAT 重复建议 | 4 | 0.7% |
| ERR-PRIVACY 隐私追问 | 3 | 0.5% |
| ERR-DIAGNOSE 诊断化表达 | 1 | 0.2% |
| **无错误** | 268 | **48.7%** |

**含错误样本：282/550（51.3%）** | **平均质量分 3.60/5**

## 四、核心发现

### 基线主要失败模式（LLM评估，按严重度）

| 优先级 | 错误 | 占比 | 说明 |
|---|---|---|---|
| 🟠 高 | **抢先建议** | 20.2% | 最大问题，未倾听就给建议 |
| 🟠 高 | **建议过载** | 15.6% | 一轮多条建议 |
| 🟠 高 | **策略时机不当** | 13.6% | 策略与需求不匹配 |
| 🟠 高 | **共情空泛** | 11.6% | 共情模板化，缺乏具体性 |
| 🔴 高危 | 风险漏检 | 4.2% | 用户有风险信号但未安全回应 |

### 场景专项发现

- **listen_only**（87条）：模型仍主动给建议（PREADVICE 20.2%），违背"只想被倾听"需求
- **safety_boundary**（58条）：风险漏检23条，安全回应不充分
- **reject_advice**（68条）：回复最长（373字符），但策略调整质量待验证

## 五、优先修复清单

| 排序 | 问题 | 占比 | SFT数据对策 |
|---|---|---|---|
| 1 | 抢先建议 | 20.2% | 增加"先倾听后建议"示范，强化listen_only场景 |
| 2 | 建议过载 | 15.6% | 增加"单建议"示范，控制回复长度 |
| 3 | 策略时机不当 | 13.6% | 增加场景-策略匹配训练样本 |
| 4 | 共情空泛 | 11.6% | 增加具体共情样本，专家共情做示范 |
| 5 | 风险漏检 | 4.2% | 强化安全边界样本，高风险专项SFT |

## 六、基线参照意义

- 此基线将作为后续所有版本的**回归对比基准**
- SFT后需复测，对比错误率下降
- 目标：抢先建议从20.2%显著下降，平均质量分从3.60提升至≥4

## 七、产物

| 文件 | 说明 |
|---|---|
| [error_type_library_v1.md](../docs/error_type_library_v1.md) | 错误类型库V1（14类） |
| [analyze_errors.py](../evaluation/analyze_errors.py) | 规则错误检测 |
| [deep_error_analysis.py](../evaluation/deep_error_analysis.py) | DeepSeek深度错误分析 |
| reports/baselines/qwen2.5-7b-instruct/ | 550条基线输出 |
| reports/baseline_error_analysis.json | 规则分析结果 |
| reports/deep_error_analysis.jsonl | 深度分析结果 |
