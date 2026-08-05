# 压力管理后训练项目 — 全项目进度汇总

> 汇总日期：2026-08-05 | 汇总范围：步骤0-7

---

## 一、项目概览

| 项目 | 内容 |
|---|---|
| 项目类型 | 纯模型训练研究（无Agent/RAG/服务部署） |
| 目标模型 | 中文压力管理对话模型（8B级） |
| 训练主线 | QLoRA-SFT → DPO → 可选GRPO |
| 基座选择 | **Qwen2.5-7B-Instruct** |
| 数据来源 | SmileChat(中文55K) + DeepSeek合成 + ESConv/ESCoT(英文待译) |
| 硬件 | RTX 5070 Laptop 8GB / PyTorch 2.13+cu130 |

---

## 二、步骤0-7 完成状态总表

| 步骤 | 名称 | 状态 | 核心产出 |
|---|---|---|---|
| 0 | 项目范围与实验环境 | ✅ | conda环境、目录结构、smoke test通过 |
| 1 | 模型行为与安全边界 | ✅ | PRD、行为Spec、风险分级、评测rubric |
| 2 | 筛选基座模型 | ✅ | Qwen2.5-7B选定，550条基线评测 |
| 3 | 数据体系与schema | ✅ | 标签体系、4类schema、标注指南 |
| 4 | 收集公开数据 | ✅ | ESConv+Adorable+SmileChat+ESCoT |
| 5 | 专家数据设计 | ✅ | 7类难点矩阵、审校流程、金标准规范 |
| 6 | LLM合成数据 | ✅ | DeepSeek v4-flash生成5,300条 |
| 7 | 清洗去重切分 | ✅ | 42,677条清洗数据，train 36,270 |

---

## 三、详细进度

### 步骤0：项目范围与实验环境 ✅

**环境**：
- Conda环境 `stress-mgmt`（Python 3.10.20）
- PyTorch 2.13.0+cu130（适配RTX 5070 Blackwell sm_120）
- GPU: RTX 5070 Laptop 8GB

**关键踩坑**：
1. RTX 5070需cu130版PyTorch（cu126不兼容）
2. conda环境缺SSL证书，需从certifi复制
3. vLLM/SGLang不支持Windows（路径过长）
4. 终端GBK编码不支持emoji，脚本用ASCII标记

**产出**：目录结构、requirements.txt、配置驱动训练脚本、smoke test（6步全通过）

---

### 步骤1：模型行为与安全边界 ✅

**核心文档**：
- PRD：6大核心能力 + 7项绝对禁止
- 风险分级L0-L3：触发条件/允许行为/禁止行为/升级路径
- Safety Spec：7条硬规则过滤 + 16个红队场景
- Model Behavior Spec：8项正向(C1-C8) + 18项负向(N1-N18)行为
- 训练评测Rubric：7维度1-7分 + 5级偏好比较 + RL奖励定义

---

### 步骤2：筛选基座模型 ✅

**决策**：Qwen2.5-7B-Instruct为主线（8GB可稳定训练）

**评测集**：550条 × 6场景类型（common_stress/listen_only/ask_for_plan/multi_turn/reject_advice/safety_boundary）

**基线结果**（Qwen2.5-7B零样本）：
- 550/550通过，0失败
- listen_only回复最短(95字)——模型会倾听
- reject_advice回复最长(373字)——认真调整策略
- 14B跳过：8GB显存不足

---

### 步骤3：数据体系与统一schema ✅

**标签体系**：10领域 × 6用户目标 × 5严重度 × 11策略 × 9错误标签

**4类schema**：SFT单轮/多轮、偏好对、独立评测

**标注指南**：就高不就低、多标签、SV-RSK强制双人标注

---

### 步骤4：收集公开数据 ✅

**已获取**：
| 数据集 | 规模 | 语言 | 来源 |
|---|---|---|---|
| SmileChat | 55,120 | 中文 | GitHub git clone |
| ESCoT | 1,707 | 英文 | GitHub git clone |
| ESConv | 910 | 英文 | HF镜像 |
| Adorable EQ | 170 | 中文 | HF镜像 |

**待获取**：CPsyCounD(3.1K)、SoulChat(258K)——网络受限

---

### 步骤5：专家数据设计 ✅（文档就绪，待专家资源）

- 7类难点覆盖矩阵（只想倾听/多压力源/信息不完整/拒绝建议/方法无效/目标变化/安全边界）
- 专家资质分层 + 审校流程 + 金标准集规范

---

### 步骤6：LLM合成数据 ✅

**DeepSeek v4-flash 生成**：
- 5,000条（POC 300条 + 批量5,000条）
- 成本极低：5,300条约10元
- 81-83% JSON成功率

**关键踩坑**：API响应截断（max_tokens=512不足）、中文JSON键映射

---

### 步骤7：清洗去重切分 ✅

**清洗管道**（clean_data.py）：
- 格式/角色/语言/长度校验 + 红线过滤（诊断/药物/电话/依赖/编造机构）

**去重优化**：MinHash O(n²) → **LSH**（6万条4分钟）

**最终切分**：
```
原始: 63,286 → 清洗: 42,677
  train: 36,270 (85%)
  dev:    4,262 (10%)
  test:   2,145 (5%)  ← 冻结，0泄漏
```

**数据池超MVP目标**（20k-30k）！

---

## 四、数据池最终构成

```
SmileChat(中文):   37,305   (87%)
DeepSeek合成:       5,372   (13%)
合计(清洗后):      42,677
```

**领域分布**：DS-LRN 12,012 > DS-INT 8,004 > DS-REL 7,655 > DS-WRK 6,547 > DS-FAM 5,859 > 其他少量

---

## 五、踩坑汇总（跨步骤）

| # | 坑 | 解决 |
|---|---|---|
| 1 | RTX 5070 Blackwell 需cu130 PyTorch | pip装2.13+cu130 |
| 2 | conda环境缺SSL证书 | 从certifi复制 |
| 3 | Windows不支持vLLM/flash-attn | 跳过，用sdpa |
| 4 | GBK终端不支持emoji | 脚本用ASCII标记 |
| 5 | HuggingFace网络受限 | hf-mirror/git clone/手动下载 |
| 6 | VPN下Python HTTP库连接重置 | 浏览器可下，命令行不行 |
| 7 | 7B作教师JSON不稳定 | 换DeepSeek API |
| 8 | API响应截断 | 增大max_tokens |
| 9 | MinHash O(n²)太慢 | 改LSH |
| 10 | 大文件超GitHub 100MB | git排除，脚本可重生成 |

---

## 六、下一步（步骤8-14）

| 步骤 | 名称 | 说明 |
|---|---|---|
| 8 | 训练前基线 | 在评测集跑基线，建错误类型库 |
| 9 | SFT-Stage1 | 领域行为对齐（QLoRA 4-bit） |
| 10 | SFT-Stage2 | 多轮与安全专项 |
| 11 | 偏好数据 | 3-6候选/上下文 |
| 12 | DPO/ORPO | 偏好优化 |
| 13 | 可选GRPO | 强化学习 |
| 14 | 全面评测 | 自动+人工+消融 |

**下一步优先**：步骤8训练前基线（数据已就绪）
