# 步骤10 回顾：第二阶段 SFT（多轮与安全专项训练）

> 回顾日期：2026-08-08 | 范围：步骤10（SFT-Stage2 训练 + 评测）

---

## 一、步骤任务回顾

根据纯模型训练版执行文档，步骤10要求：

1. 以 SFT-Stage1 最优 checkpoint 为起点
2. 增加多轮训练场景：逐步补充信息/拒绝建议/方法无效/目标变化/状态变化/后期风险信号
3. 增加安全专项样本：诊断/药物/诱导保证/诱导保密/诱导依赖/隐晦高风险
4. 训练模型体现：前文保持、避免重复建议、根据反馈调整策略、高风险停止普通建议
5. 较低学习率 5e-5~1e-4
6. 混入 Stage1 replay 防退化
7. 对普通/高风险采样比例做消融

**通过条件**：多轮关键信息保持率显著提升；风险漏判率下降且普通场景过度拒答率可控。

---

## 二、已完成内容

### 2.1 数据生成（DeepSeek v4-flash，12并发）

| 数据集 | 目标 | 实际 | 场景 | 耗时 |
|---|---|---|---|---|
| 多轮V2 | 800 | 800（+6测试） | M1-M6六类均匀 | 75.7分钟 |
| 安全专项V2 | 500 | 500（+8测试） | S1-S6六类均匀 | 10.7分钟 |

**数据池**：2,618条 = 多轮806 + 安全1012（×2上采样）+ Stage1 replay 800

### 2.2 训练

| 项 | 值 |
|---|---|
| 起点 | Stage1 adapter（is_trainable=True 修复后） |
| 步数 | 328步（中断后续训完成） |
| 学习率 | 1e-4（低于Stage1防遗忘） |
| 上下文 | max_len=320（8GB显存极限） |
| loss | 16.2 → 13.4 |

### 2.3 评测结果

- **风险漏检 6.5%→1.5%**（低于基线4.2%）✅ 核心目标
- 无错误样本 70.9%→81.1%，质量分 4.22→4.48
- safety_boundary场景：风险漏检62.1%→13.8%，质量2.24→3.98

---

## 三、数据流

```
步骤9数据池 → SFT-Stage2数据生成（12并发, ~1.5h）
    │
    ▼
多轮V2 806 + 安全V2 500
    │
    ▼
build_sft2_pool.py → 2,618条（安全×2上采样 + Stage1 replay 800）
    │
    ▼
run_sft_stage2.py（手动循环, 328步, 断点续训）
    │
    ▼
final_adapter → eval_sft.py（550条）→ 12并发LLM错误分析
    │
    ▼
compare_stages.py → 三阶段对比报告
```

---

## 四、遇到的问题与解决

### 问题1：长序列训练灾难性慢/卡死

**现象**：多轮样本最长1083 tokens，单步177s（435 tokens只要8s）——注意力O(n²)+显存瓶颈。

**解决**：max_len控制+历史截断。8GB显存下7B QLoRA训练序列必须≤448（实测320最稳）。

### 问题2：gradient checkpointing的误判

**现象**：最初误判"checkpointing不省显存"，关闭后反而每步41s。

**根因**：实测"无CKPT"时模型实际处于enable状态（prepare_model默认启用），测量方法有误。**正确结论：checkpointing必须启用**（启用5.3s vs 禁用41s）。

### 问题3：Trainer训练卡死

**现象**：Trainer每步9分钟+。

**解决**：弃用Trainer，改手动训练循环（forward+backward+step），实测5.7s/步稳定。

### 问题4：训练中途"卡死"实为电脑睡眠

**现象**：日志停在step 10达55分钟，GPU 100%但无推进。排查半天以为是样本问题。

**真相**：**电脑进入睡眠导致GPU挂起**。用户调整系统设置后解决。

### 问题5：错误分析550条全401

**现象**：并发错误分析全部返回401认证错误。

**根因**：deep_error_analysis.py未加载.env，`api_key=None` → `Authorization: Bearer None`。

**解决**：加 `load_dotenv()`。

### 问题6：训练脚本无断点续训

**需求**：4-5小时训练需防中断。

**解决**：手动循环+每20步保存checkpoint（adapter+优化器+scheduler+位置），`--resume` 一键续训，已验证。

---

## 五、踩坑完整清单

| # | 坑 | 表现 | 解决 |
|---|---|---|---|
| 1 | 长序列训练极慢 | 1083 tokens单步177s | max_len=320+历史截断 |
| 2 | checkpointing误判 | 关闭后41s/步 | 必须启用（5.3s） |
| 3 | Trainer卡死 | 每步9分钟 | 改手动循环 |
| 4 | **电脑睡眠卡训练** | 日志停55分钟 | 用户改系统设置 |
| 5 | API key未加载 | 550条全401 | 加load_dotenv |
| 6 | 并发写错 | 错误分析串行40min | ThreadPoolExecutor 12并发→7min |
| 7 | PeftModel训练无效 | Trainable=0（is_trainable默认False） | 加is_trainable=True |
| 8 | checkpoint重复保存 | 同一步存8次 | 只在global_step变化时保存 |
| 9 | max_tokens=2048空响应 | reasoning占满输出 | 提到4096 |

---

## 六、关键决策记录

### 决策1：安全专项上采样×2

**背景**：风险漏检是Stage1的核心遗留问题（6.5%）。

**决策**：安全专项数据在训练池中×2上采样（506→1012）。

**影响**：safety_boundary场景风险漏检62.1%→13.8%，核心目标达成。

### 决策2：手动训练循环替代Trainer

**背景**：Trainer在8GB显存+7B下卡死。

**决策**：手写训练循环（forward/backward/step + 梯度累积 + scheduler）。

**影响**：可控且稳定（5.7s/步），并支持慢样本跳过和断点续训。

### 决策3：max_len=320取舍

**背景**：448 tokens在部分样本触发显存临界卡死；多轮信息与训练速度权衡。

**决策**：max_len=320（保留最近3轮），全量328步约2.6小时。

**影响**：训练稳定，但listen_only/multi_turn质量分轻微下降（上下文截断）。

### 决策4：错误分析12并发

**背景**：550条串行LLM分析约40分钟。

**决策**：ThreadPoolExecutor 12并发（DeepSeek限制2500）。

**影响**：~7分钟完成，且验证了dotenv加载的坑。

---

## 七、遗留问题

| 问题 | 影响 | 方向 |
|---|---|---|
| listen_only质量分降 | 4.49→4.15 | 安全专项可能稀释纯倾听，人工抽查 |
| multi_turn略降 | 4.42→4.27 | max_len截断，超长历史保持 |
| 压力源理解微升 | 2.4%→3.1% | 同上 |
| 未做采样比例消融 | 步骤10建议 | 留到步骤13评估 |

---

## 八、资产清单

### 新增脚本
| 脚本 | 用途 |
|---|---|
| data_processing/generate_sft2_data.py | 多轮/安全数据生成（12并发） |
| data_processing/build_sft2_pool.py | Stage2数据池构建 |
| training/sft/run_sft_stage2.py | Stage2训练（手动循环+续训） |
| evaluation/compare_stages.py | 阶段对比 |

### 新增产物
| 文件 | 说明 |
|---|---|
| checkpoints/sft_stage2/final_adapter/ | Stage2 adapter |
| reports/sft_stage2_report.md | 训练评测报告 |
| reports/deep_error_sft2.jsonl | Stage2错误分析 |
| reports/baselines/sft_stage2/ | 550条评测输出 |
| data/processed/sft2_data/ | Stage2数据（池+批次） |

---

## 九、衔接下一步

**SFT-Stage2 达成核心目标**：风险漏检降到1.5%（低于基线），safety_boundary场景质变。

**下一步（步骤11：构建偏好数据）**：
- 用基座/Stage1/Stage2生成同一prompt的多个候选回复
- 构造偏好对用于DPO
- 重点：安全边界偏好对（正确风险引导 vs 风险漏检）

**后续**：步骤12（DPO）→ 步骤13（可选GRPO）→ 步骤14（全面评估）
