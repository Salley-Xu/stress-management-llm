# 公开数据目录与许可审计

> 版本：V1.0 | 日期：2026-08-04 | 对应步骤4

---

## 1. 候选数据集总览

### 1.1 中文数据集（优先使用）

| 数据集 | 规模 | 类型 | 许可 | 质量评估 | 优先级 |
|---|---|---|---|---|---|
| **SmileChat / MeChat** | 55K 多轮对话 | 情绪支持 | Apache-2.0 | 基于真实PsyQA扩展，质量较高 | ⭐⭐⭐ |
| **SoulChat** | 258K 对话 | 共情对话 | 待确认 | 规模大，需抽检质量 | ⭐⭐ |
| **CPsyCounD** | 3.1K 多轮 | 心理咨询 | CC-BY | 从真实咨询报告构建，质量高 | ⭐⭐⭐ |
| **ESCoT** | 1.7K 多轮+CoT | 情绪支持 | 待确认 | ACL 2024，有策略推理标注 | ⭐⭐⭐ |
| **Adorable EQ Chat** | 170 对话组 | 高情商对话 | CC-BY-4.0 | 规模小，风格参考 | ⭐ |

### 1.2 英文数据集（需翻译）

| 数据集 | 规模 | 类型 | 许可 | 质量评估 | 优先级 |
|---|---|---|---|---|---|
| **ESConv** | 1.3K 多轮 | 情绪支持 | Apache-2.0 | 标准benchmark，有策略标签 | ⭐⭐ |
| **ESConv-SRA** | 25K-42K 续写 | 长程情绪支持 | 待确认 | LLaMA3-70B生成续写 | ⭐ |

---

## 2. 数据集详细审计

### 2.1 SmileChat / MeChat

| 属性 | 值 |
|---|---|
| HuggingFace | `qiuhuachuan/MeChat` |
| 论文 | SMILE: Single-turn to Multi-turn Inclusive Language Expansion (2024) |
| 规模 | 55,165 条多轮对话 |
| 来源 | 基于真实PsyQA数据使用ChatGPT扩展为多轮 |
| 语言 | 中文 |
| 许可 | Apache-2.0 |
| 可用于训练 | ✅ |
| 可修改 | ✅ |
| 可再分发 | ✅ |
| 含PII | 否（已脱敏） |
| 含临床内容 | 部分（需过滤） |
| 适合本项目 | ✅（需过滤诊断性表达和药物建议） |

**建议处理**：
- 过滤包含临床诊断术语的样本
- 过滤包含药物推荐的样本
- 保留倾听、情绪确认、问题拆解类型对话
- 预计可保留 70%–80%

### 2.2 CPsyCounD

| 属性 | 值 |
|---|---|
| 论文 | CPsyCounD: A Report-based Multi-turn Dialogue Reconstruction and Evaluation Framework (ACL 2024 Findings) |
| 规模 | 3,145 条多轮对话 |
| 来源 | 从真实心理咨询报告重建（Memo2Demo方法） |
| 语言 | 中文 |
| 许可 | CC-BY |
| 可用于训练 | ✅ |
| 可修改 | ✅ |
| 可再分发 | ✅ |
| 含PII | 否（来自公开报告，已脱敏） |
| 含临床内容 | 部分（心理咨询场景，需谨慎过滤） |
| 适合本项目 | ✅（高质量，但需要过滤正式咨询场景中不适合本模型边界的内容） |

**建议处理**：
- 本模型不做心理咨询，需过滤正式咨询框架的表达
- 保留压力管理、情绪支持部分
- 预计可保留 50%–60%

### 2.3 ESCoT

| 属性 | 值 |
|---|---|
| GitHub | `TeigenZhang/ESCoT` |
| 论文 | ESCoT: Towards Interpretable Emotional Support Dialogue Systems (ACL 2024) |
| 规模 | 1,700+ 条对话 + Chain-of-Thought 标注 |
| 来源 | 人工标注 + LLM辅助生成 |
| 语言 | 中文 |
| 许可 | MIT |
| 可用于训练 | ✅ |
| 策略标签 | 14种策略（可直接映射到我们的11种策略） |
| 含PII | 否 |
| 含临床内容 | 少量 |
| 适合本项目 | ✅（有策略标签，可直接用于SFT训练） |

**建议处理**：
- 策略标签质量高，可优先保留
- CoT推理链可用于增强模型推理能力

### 2.4 SoulChat

| 属性 | 值 |
|---|---|
| 规模 | ~258K 条对话 |
| 来源 | LLM合成 |
| 许可 | 待确认（HuggingFace `scutcyr/SoulChat`） |
| 质量 | 合成数据，需大规模抽检 |
| 适合本项目 | ✅（规模大，可作为补充） |

### 2.5 ESConv (英文)

| 属性 | 值 |
|---|---|
| HuggingFace | `thu-coai/esconv` |
| 规模 | 1,300 条多轮对话 |
| 来源 | 人工编写 |
| 许可 | Apache-2.0 |
| 策略标签 | 8种策略 |
| 适合本项目 | ✅（经典benchmark，需翻译为中文） |

---

## 3. 数据红线过滤清单

所有入库数据必须过滤以下内容：

| 过滤项 | 检测方式 | 处理 |
|---|---|---|
| 诊断性表述 | 关键词匹配 + LLM分类 | 删除含"抑郁症"、"焦虑症"等诊断标签的回复 |
| 药物推荐 | 关键词匹配 + LLM分类 | 删除含具体药名的回复 |
| 热线/机构名 | 正则匹配电话号码/机构名模式 | 删除或替换为占位符 |
| 极端依赖表达 | LLM分类 | 删除"你是我唯一的希望"等 |
| 操控性表达 | LLM分类 | 删除威胁、操控用户的内容 |
| PII | regex + presidio | 脱敏处理 |
| 乱码/截断 | 规则检查 | 删除 |

---

## 4. 格式转换计划

所有数据集统一转换为 [data_schema_v1.md](data_schema_v1.md) 定义的 SFT schema：

| 数据集 | 原始格式 | 转换脚本 | 状态 |
|---|---|---|---|
| MeChat | Parquet/JSON | data_processing/convert_mechat.py | 待开发 |
| CPsyCounD | JSON | data_processing/convert_cpsycd.py | 待开发 |
| ESCoT | JSON | data_processing/convert_escot.py | 待开发 |
| SoulChat | JSON | data_processing/convert_soulchat.py | 待开发 |
| ESConv | JSON | data_processing/convert_esconv.py | 待开发 |

---

## 5. 非中文数据处理策略

| 策略 | 成本 | 质量 | 适用 |
|---|---|---|---|
| 人工翻译 | 高 | 最高 | ESConv（~1.3K条，可接受） |
| LLM翻译 + 人工抽检 | 中 | 较高 | ESConv-SRA（规模大） |
| LLM翻译 + 术语修正 | 低 | 中等 | 快速原型验证 |

---

## 6. 已下载数据状态（2026-08-05 更新）

### 6.1 已获取数据集

| 数据集 | 规模 | 语言 | 许可 | 转换状态 |
|---|---|---|---|---|
| ESConv-LLM | 910条 | 英文 | Apache-2.0 | ✅ 已转schema (esconv_sft_v1.jsonl) |
| Adorable EQ | 170条 | 中文 | CC-BY-4.0 | ✅ 已转schema (adorable_eq_sft_v1.jsonl) |

### 6.2 ESConv 详细审计

| 属性 | 值 |
|---|---|
| HuggingFace | `Estwld/esconv_llm`（源自 `thu-coai/esconv`） |
| 规模 | 910条多轮对话（train） |
| 平均轮数 | 18-29轮 |
| 策略标签 | 8种（Question/Restatement/Reflection/Affirmation/Suggestions/Information/Self-disclosure/Others） |
| 情感标注 | anxiety/depression/loneliness/sadness/fear/anger等 |
| 许可 | Apache-2.0（可用于训练） |
| 待处理 | **需翻译为中文** |

### 6.3 待获取数据集（网络受限）

以下数据集当前无法通过镜像获取，保留在待获取清单：

| 数据集 | 规模 | 优先恢复 |
|---|---|---|
| MeChat/SmileChat | 55K | ⭐⭐⭐ |
| CPsyCounD | 3.1K | ⭐⭐⭐ |
| ESCoT | 1.7K | ⭐⭐⭐ |
| SoulChat | 258K | ⭐⭐ |

**恢复方案**：尝试GitHub直连（需VPN网络环境）、ModelScope镜像、或作者提供的国内网盘。

---

## 7. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| V1.0 | 2026-08-04 | 初稿，5个候选数据集审计 |
| V1.1 | 2026-08-05 | 补充已下载数据状态（ESConv 910 + Adorable 170） |
