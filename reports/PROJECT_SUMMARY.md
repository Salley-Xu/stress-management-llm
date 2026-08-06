# 压力管理后训练项目 — 训练前阶段完整回顾

> 汇总日期：2026-08-06 | 范围：步骤0-7（数据准备）+ 步骤8-9（基线/SFT已推进）
> 说明：本文档重点回顾训练前的数据准备工作，供后续阶段参考。

---

## 一、项目概览

| 项目 | 内容 |
|---|---|
| 项目类型 | 纯模型训练研究（无Agent/RAG/服务部署） |
| 目标模型 | 中文压力管理对话模型（8B级） |
| 训练主线 | QLoRA-SFT → DPO → 可选GRPO |
| 基座选择 | **Qwen2.5-7B-Instruct** |
| 硬件 | RTX 5070 Laptop 8GB / PyTorch 2.13+cu130 |
| 当前进度 | 步骤9（SFT-Stage1）已完成 |

---

## 二、步骤0-7 完成状态总表

| 步骤 | 名称 | 状态 | 核心产出 |
|---|---|---|---|
| 0 | 项目范围与实验环境 | ✅ | conda环境、目录结构、smoke test通过 |
| 1 | 模型行为与安全边界 | ✅ | PRD、行为Spec、风险分级、双Rubric |
| 2 | 筛选基座模型 | ✅ | Qwen2.5-7B选定，550条基线评测 |
| 3 | 数据体系与schema | ✅ | 标签体系、4类schema、标注指南 |
| 4 | 收集公开数据 | ✅ | SmileChat 55K + ESCoT 1.7K + ESConv + Adorable |
| 5 | 专家数据 | ✅ | DeepSeek生成503条专家数据（Rubric筛选） |
| 6 | LLM合成数据 | ✅ | DeepSeek生成5,300条 |
| 7 | 清洗切分+质量评估 | ✅ | 质量评估发现SmileChat问题 → 混合策略 → 8,868条 |

---

## 三、各步骤详细回顾

### 步骤0：项目范围与实验环境 ✅

**环境**：Conda `stress-mgmt`（Python 3.10）、PyTorch 2.13+cu130、RTX 5070 8GB

**产出**：目录结构、requirements.txt、配置驱动训练脚本、QLoRA smoke test（6步全通过）

**踩坑**：
1. RTX 5070 (Blackwell sm_120) 需 **cu130版PyTorch**，cu126不兼容
2. conda环境缺SSL证书 → 从certifi复制
3. Windows不支持vLLM/flash-attn → 用sdpa
4. 终端GBK编码不支持emoji → 脚本用ASCII标记

---

### 步骤1：模型行为与安全边界 ✅

**核心文档**：
- PRD（6大核心能力 + 7项绝对禁止）
- 风险分级L0-L3（触发/允许/禁止/升级）
- Safety Spec（7条硬规则 + 16个红队场景）
- Model Behavior Spec（8正向 + 18负向行为）
- **双Rubric**：35分制专家Rubric + 100分制训练数据质量标准

**说明**：用户设计了两个Rubric体系，分别用于专家数据审校（35分制）和训练数据筛选（100分制）。

---

### 步骤2：筛选基座模型 ✅

**决策**：Qwen2.5-7B-Instruct为主线（8GB可稳定训练）

**评测集**：550条 × 6场景（common_stress/listen_only/ask_for_plan/multi_turn/reject_advice/safety_boundary）

**基线**：550/550通过，0失败；listen_only回复最短(95字)；reject_advice最长(373字)

**踩坑**：
- HuggingFace网络受限 → hf-mirror/git clone/手动下载多种途径
- VPN下Python HTTP库连接重置（浏览器却可以）

---

### 步骤3：数据体系与统一schema ✅

**标签体系**：10领域 × 6用户目标 × 5严重度 × 11策略 × 9错误标签
**4类schema**：SFT单轮/多轮、偏好对、独立评测

---

### 步骤4：收集公开数据 ✅

**已获取**：
| 数据集 | 规模 | 语言 | 获取方式 |
|---|---|---|---|
| SmileChat | 55,120 | 中文 | GitHub git clone |
| ESCoT | 1,707 | 英文 | GitHub git clone |
| ESConv | 910 | 英文 | HF镜像 |
| Adorable EQ | 170 | 中文 | HF镜像 |

**踩坑**：
- hf-mirror.com需在**shell层export** HF_ENDPOINT（Python内设置无效）
- 中文大数据集（MeChat/CPsyCounD等）在镜像不可用 → git clone解决SmileChat/ESCoT

---

### 步骤5：专家数据 ✅

**执行**：用户设计Rubric（35分制：7维度×0-5分）后，用DeepSeek按专家角色生成503条专家数据。

**Rubric筛选**：核心(≥28) 479条 + 普通(24-28) 24条，平均33.3/35

**7类难点覆盖**：E1倾听/E2多压力源/E3信息不全/E4拒绝建议/E5方法无效/E6目标变化/E7安全边界

**踩坑**：
- DeepSeek `response_format:json_object` 与长prompt导致空响应 → 移除
- 多轮JSON截断（max_tokens=1024不足）→ 提高到2048
- 偶发空响应 → 加重试逻辑
- 标签规范化（Rubric风格 → 项目DS-/UG-/ST-标准）

---

### 步骤6：LLM合成数据 ✅

**执行**：DeepSeek v4-flash 生成5,300条（POC 300 + 批量5,000）

**成本**：约10元（极低，v4-flash是当前性价比最高模型）

**踩坑**：
- 本地7B作教师JSON不稳定 → 换DeepSeek API
- API响应截断 → 提高max_tokens
- 中文JSON键（用户/助手）→ 解析映射

---

### 步骤7：清洗切分 + 质量评估 ✅（关键转折点）

**清洗**：63,286 → 42,677条（格式/角色/语言/长度校验 + 红线过滤 + 去重）

**关键转折**：用**100分制Rubric**评估300条抽样，发现：

| 来源 | 合格率(≥70分) |
|---|---|
| 专家数据 | **99%** |
| DeepSeek合成 | **98%** |
| **SmileChat** | **8.5%** ⚠️ |

**SmileChat虽然占86%数据量，但质量不合格**（回复模板化、说教、理解浅）。

**处理决策（混合策略）**：专家+合成全保留（5,868条），SmileChat只抽样3,000 → **最终8,868条**

**踩坑**：
- MinHash去重O(n²)在6万条太慢 → 改**LSH**（4分钟完成）
- 大文件超GitHub 100MB限制 → git排除，脚本可重生成

---

## 四、数据质量评估的深层教训

### 4.1 "数量优先"陷阱

最初以SmileChat（55K）为主扩充数据池，但质量评估发现**91%不合格**。教训：
- **数据规模不等于数据质量**
- LLM合成（98%合格）和专家数据（99%合格）远优于真实但低质量的SmileChat

### 4.2 为什么SmileChat不合格

- 基于真实PsyQA用LLM扩展，但counselor回复**模板化、说教**
- 不适合"压力管理支持助手"的对话风格
- 质量评估（100分制）精准识别了这个问题

### 4.3 数据策略转变

从"数量优先"转为**"质量优先"**：
- 核心训练数据：专家+合成（5,868条高质量）
- SmileChat降为补充（抽样3,000）
- 后续如需扩充，优先合成/专家而非SmileChat

---

## 五、踩坑完整清单（跨步骤）

| # | 坑 | 解决 |
|---|---|---|
| 1 | RTX 5070 Blackwell 需cu130 PyTorch | pip装2.13+cu130 |
| 2 | conda环境缺SSL证书 | 从certifi复制 |
| 3 | Windows不支持vLLM/flash-attn | 跳过，用sdpa |
| 4 | GBK终端不支持emoji | 脚本用ASCII标记 |
| 5 | HuggingFace网络受限 | hf-mirror/git clone/手动下载 |
| 6 | VPN下Python HTTP连接重置 | 浏览器可下，命令行不行 |
| 7 | hf-mirror需shell层export | 非Python内设置 |
| 8 | 7B作教师JSON不稳定 | 换DeepSeek API |
| 9 | DeepSeek json_object空响应 | 移除response_format |
| 10 | API响应截断 | 增大max_tokens |
| 11 | 中文JSON键解析 | 键映射 |
| 12 | MinHash O(n²)太慢 | 改LSH |
| 13 | 大文件超GitHub 100MB | git排除，脚本重生成 |
| 14 | **SmileChat质量陷阱** | 100分制评估发现，混合策略处理 |

---

## 六、训练数据最终构成（步骤9 SFT用）

```
专家数据:     491  (99%合格，Rubric 33.3/35)
DeepSeek合成: 2,500 (抽样，98%合格)
─────────────────────────────
SFT子集:     2,991 (高质量)
```

**SFT-Stage1结果**：质量分3.60→4.22（+0.62），抢先建议↓89%，达到发布门槛。

---

## 七、脚本与文档资产

### 数据管道脚本

| 脚本 | 功能 |
|---|---|
| data_processing/convert_smilechat.py | SmileChat→schema |
| data_processing/convert_escot.py | ESCoT→schema |
| data_processing/convert_esconv.py | ESConv→schema |
| data_processing/generate_synthetic.py | DeepSeek合成 |
| data_processing/generate_expert_data.py | 专家数据+35分Rubric筛选 |
| data_processing/normalize_expert_labels.py | 标签规范化 |
| data_processing/clean_data.py | 清洗去重(LSH) |
| data_processing/split_data.py | 分层切分+泄漏审计 |
| data_processing/build_final_pool.py | 混合策略重建 |
| data_processing/evaluate_training_data.py | 100分制质量评估 |

### 关键文档

| 文档 | 说明 |
|---|---|
| docs/expert_rubric.md | 35分制专家Rubric |
| docs/training_data_quality_standard.md | 100分制质量标准 |
| docs/judge_prompts.md | LLM-as-a-Judge prompt模板 |
| docs/data_taxonomy_v1.md | 标签体系 |
| reports/data_quality_report.md | 质量评估报告 |

---

## 八、下一步

- **SFT-Stage2**：多轮与安全专项（重点解决风险漏检↑问题）
- **DPO偏好优化**
- 数据扩充方向：优先合成/专家，补齐安全边界样本
