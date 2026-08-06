# 步骤9 回顾：第一阶段 SFT（领域行为对齐）

> 回顾日期：2026-08-06 | 范围：步骤9（SFT-Stage1 训练 + 评测）

---

## 一、步骤任务回顾

根据纯模型训练版执行文档，步骤9包含9项任务：

1. 指令模型为起点，QLoRA 训练（4-bit + BF16 + LoRA rank 32/64 + alpha 64/128 + LR 1e-4~2e-4 + epoch 1-3 + context 4k + gradient checkpointing）
2. 以高质量单轮和短多轮样本为主
3. 数据混合：公开25-40% + 专家20-30% + LLM合成30-50%
4. 混入10-15%通用指令数据防遗忘
5. 专家/安全边界/困难样本提高采样权重
6. 保留简短自然回复，避免统一长篇结构化
7. 每checkpoint在开发集评测并按综合指标早停
8. 消融：LoRA rank 32 vs 64、数据量、有无通用数据、混合 vs 仅合成
9. 产出 checkpoint + 训练日志 + 消融报告 + 错误变化报告

**通过条件**：压力理解、共情和建议质量显著优于基座；安全边界和通用能力无明显退化。

---

## 二、已完成内容

### 2.1 训练设置（实际执行）

| 项 | 值 |
|---|---|
| 基座模型 | Qwen2.5-7B-Instruct |
| 训练数据 | **2,991条高质量子集**（专家491 + 合成2,500） |
| 方法 | QLoRA 4-bit nf4, rank=32, alpha=64, dropout 0.05 |
| 目标模块 | 全部7个线性层（q/k/v/o/gate/up/down） |
| 可训练参数量 | 80.7M（1.82%） |
| 训练步数 | 374步（1 epoch） |
| 有效batch | 8（batch=1 × grad_accum=8） |
| 学习率 | 1.5e-4 (cosine, warmup 5%) |
| 精度 | BF16 + TF32 + gradient checkpointing |
| 训练耗时 | **128.5分钟（2.14小时）** |
| 最终train_loss | 1.812 |

### 2.2 评测设置

| 项 | 值 |
|---|---|
| 评测集 | 550条（6场景，与基线相同，已冻结） |
| 解码 | temperature=0.1, do_sample=False, max_new_tokens=512 |
| 错误分析 | DeepSeek LLM（17类错误 + 1-5分质量评分） |

### 2.3 核心结果

| 指标 | 基线 | SFT后 | 变化 |
|---|---|---|---|
| **平均质量分** | 3.60/5 | **4.22/5** | **↑ +0.62** ✅ |
| 无错误样本 | 48.7% | **70.9%** | **↑ +22.2%** |
| 抢先建议 | 20.2% | 2.2% | **↓ 89%** ✅ |
| 建议过载 | 15.6% | 0.2% | **↓ 99%** ✅ |
| 共情空泛 | 11.6% | 1.6% | **↓ 86%** ✅ |
| 策略时机不当 | 13.6% | 9.6% | ↓ 30% |
| **风险漏检** | 4.2% | **6.5%** | **↑ 55%** ⚠️ |
| 平均回复长度 | 221.7字符 | 68.6字符 | ↓ 69% ⚠️ |

---

## 三、训练数据流

```
步骤7混合池 8,868条（专家491 + 合成5,377 + SmileChat抽样3,000）
    │
    ▼ 质量优先取舍
SFT子集 2,991条（专家491 + 合成2,500）← 只保留最高质量两来源
    │
    ▼
run_sft_stage1.py（QLoRA, 374步, 128.5min）
    │
    ▼
final_adapter → eval_sft.py（550条评测集生成）
    │
    ▼
deep_error_analysis.py（LLM错误分析）→ 错误变化对比
```

> **关键取舍**：SFT只用专家+合成两来源（99%/98%合格），SmileChat抽样3,000条**未进训练集**——8.5%合格率数据即使抽样也不如去掉。这与步骤7"混合策略保留3K"不同：那一步是为了扩充数据池总量，SFT阶段进一步收紧。

---

## 四、遇到的问题与解决

### 问题1：trl 1.9.2 兼容性问题（训练框架被迫更换）

- **现象**：`SFTTrainer` 不接受 `max_seq_length`/`packing`/`tokenizer` 参数；与 datasets 5.x 存在 truncate bug
- **解决**：改用**标准 Trainer** + 自定义预处理（assistant掩码）+ 自定义 data_collator
- **教训**：trl 版本锁定导致 API 变动，成熟项目应优先标准 Trainer 或锁定 trl 版本

### 问题2：DataCollatorForLanguageModeling batch>1 报错

- **现象**：transformers 5.14.1 下 batch>1 时 "labels nesting" 错误（padding bug）
- **解决**：替换为**自定义 data_collator**——手动按 batch 内最大长度 pad input_ids 和 labels
- **教训**：新版本 transformers collator 行为不可靠，显式手动 pad 最可控

### 问题3：8GB 显存压力与训练速度

- **现象**：batch=4/8 时显存交换导致更慢；batch=1 单样本前向+反向约18s（7B QLoRA 在 8GB 的硬件极限）
- **解决**：batch=1 + grad_accum=8（有效batch=8），通过**控制数据量**（2,991条）把总时长压到2.14小时
- **教训**：硬件极限无法绕过，只能靠数据量/步数管理总时间

### 问题4：`max_steps=None` 触发 `_validate_args` 错误

- **现象**：transformers Trainer 比较 None 与 int 失败
- **解决**：`max_steps = args.max_steps if args.max_steps is not None else -1`
- **教训**：Trainer 对 None 参数的内部校验脆弱，传 -1 表示"不限制"更稳

### 问题5：评测阶段发现的模型新问题

- **风险漏检 4.2%→6.5%**：训练数据中安全边界样本不足，SFT后模型更倾向"共情式回应"而漏掉风险信号
- **回复过短**（221.7→68.6字符）：训练数据偏简洁，部分场景丢失共情细节
- **压力源理解错误 1.1%→2.4%** + **资源幻觉 0→0.5%**：回复变短导致信息捕获减少、边界知识未被强化

---

## 五、踩坑完整清单

| # | 坑 | 表现 | 解决 |
|---|---|---|---|
| 1 | trl 1.9.2 SFTTrainer 不兼容 | 参数不被接受、truncate bug | 换标准 Trainer |
| 2 | DataCollator batch>1 padding bug | "labels nesting" 错误 | 自定义 data_collator |
| 3 | batch>1 显存交换 | 8GB 上反而更慢 | batch=1 |
| 4 | 单样本 18s/step | 7B QLoRA 硬件极限 | 控制数据量(2,991) |
| 5 | max_steps=None 校验错误 | `_validate_args` 崩溃 | 改 -1 |
| 6 | 评测结果中恢复重现中文乱码 | 终端 GBK 显示 | PYTHONIOENCODING=utf-8 |
| 7 | 训练/评测模型路径硬编码 | D盘绝对路径，换机不可迁移 | 仅本项目本机用（文档注明） |

---

## 六、关键决策记录

### 决策1：只训练专家+合成（2,991条），SmileChat不进场

**背景**：步骤7混合池8,868条含SmileChat抽样3,000，但SmileChat合格率仅8.5%。

**决策**：SFT仅用专家491 + 合成2,500。

**影响**：数据纯净度高，直接促成抢先建议↓89%、共情空泛↓86%；但安全边界样本少，是风险漏检↑的诱因。

### 决策2：放弃框架消融（受硬件限制）

**背景**：执行文档要求消融 rank 32 vs 64、数据量、通用数据占比。

**决策**：单配置训练（rank=32、2,991条、无通用数据），不做消融。

**影响**：受8GB硬件 + 2小时/run 的成本限制，消融留到步骤13评估阶段用更小数据做子实验。当前阶段以"跑通+达标"为优先。

### 决策3：标准 Trainer 替代 trl

**背景**：trl 1.9.2 与 datasets 5.x 不兼容。

**决策**：标准 Trainer + 自定义 collator + assistant掩码（user tokens = -100）。

**影响**：训练更可控，但需自己实现 completion-masked loss。

---

## 七、评测结论

### ✅ 通过条件评估

| 通过条件 | 状态 |
|---|---|
| 压力理解/共情/建议质量显著优于基座 | ✅ 质量分+0.62，抢先建议-89% |
| 安全边界无明显退化 | ⚠️ 风险漏检↑55%（4.2%→6.5%） |
| 通用能力无明显退化 | ⚠️ 未测通用benchmark，回复偏短 |

### ⚠️ 遗留问题（SFT-Stage2 方向）

1. **风险漏检↑**：安全边界样本不足 → 需要 E7 安全专项数据
2. **回复过短**：68.6字符可能丢失共情 → 需长度校准
3. **压力源理解错误↑**：信息捕获减少 → 多轮补充信息场景

---

## 八、资产清单

### 新增脚本

| 脚本 | 用途 |
|---|---|
| training/sft/run_sft_stage1.py | SFT-Stage1 训练（标准Trainer + 自定义collator） |
| training/sft/smoke_test.py | QLoRA smoke test（步骤0产出） |
| evaluation/eval_sft.py | SFT后评测（加载adapter，550条生成） |

### 新增产物

| 文件 | 说明 |
|---|---|
| checkpoints/sft_stage1/final_adapter/ | SFT-Stage1 adapter（LoRA权重） |
| checkpoints/sft_stage1_train.log | 训练日志（374步） |
| reports/sft_stage1_report.md | SFT训练评测报告 |
| reports/deep_error_sft.jsonl | SFT后LLM错误分析 |
| reports/baselines/sft_stage1/baseline_results.jsonl | SFT后550条评测输出 |

---

## 九、衔接下一步

**SFT-Stage1 证明了两件事**：① 高质量小数据（2,991条）就能显著改善核心行为；② 安全边界数据缺失的代价是风险漏检上升。

**SFT-Stage2（步骤10）明确方向**：
- 以 final_adapter 为起点
- 补**多轮补充信息**场景（治压力源理解错误）
- 补**安全边界专项**（E7场景，治风险漏检）
- 回复长度校准（平衡简洁与共情）

**后续**：步骤11（偏好数据）→ 步骤12（DPO）→ 步骤13（可选GRPO）→ 步骤14（全面评估）
