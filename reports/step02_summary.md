# 步骤2总结：筛选基座模型

> 完成日期：2026-08-02

---

## 一、步骤任务回顾

根据《压力管理后训练项目_执行步骤（纯模型训练版）》，步骤2包含8项任务：

1. 选取2-3个中文能力较强的8B-14B开放权重模型
2. 候选至少包含8B指令模型（主线）、12B-14B指令模型（对照）、Base检查点（消融）
3. 统一所有候选的system prompt、chat template、上下文长度、解码参数
4. 构建500-1000条初始评测集，覆盖6类场景
5. 完成零样本基线测试
6. 在每个候选上使用相同2k-5k样本进行轻量QLoRA-SFT对照
7. 测试：中文自然度、压力理解、共情准确性、指令遵循、多轮一致性、安全边界、训练显存和速度
8. 按权重选择主线：任务质量35% + 安全30% + SFT可塑性20% + 训练资源15%

---

## 二、已完成内容

### 2.1 候选模型分析

| 文件 | 内容 |
|---|---|
| [docs/base_model_candidates.md](../docs/base_model_candidates.md) | 候选模型分析、选型权重、硬件约束评估 |

**主线候选**：Qwen2.5-7B-Instruct（7.6B，8GB VRAM可稳定训练）
**备选**：Qwen3-8B（如已发布且可用）
**规模对照**：Qwen2.5-14B-Instruct（仅零样本评测）
**消融对照**：Qwen2.5-7B (Base)

### 2.2 统一配置

所有候选模型的评测配置已在 [configs/fixed_params.md](../configs/fixed_params.md) 中锁定：
- System prompt 统一文本
- temperature=0.1, do_sample=False（评测用贪心解码）
- max_new_tokens=512
- 上下文长度 4096

### 2.3 评测数据集

| 场景类型 | 数量 | 说明 |
|---|---|---|
| common_stress | 186 | 常见压力场景（学习/工作/人际/家庭等） |
| listen_only | 87 | 用户只想被倾听 |
| ask_for_plan | 84 | 用户明确要求计划 |
| multi_turn | 67 | 多轮状态变化 |
| reject_advice | 68 | 建议被拒绝/策略调整 |
| safety_boundary | 58 | 安全边界场景（L0-L2） |
| **合计** | **550** | |

生成脚本：`evaluation/build_eval_set.py`，输出文件：`data/processed/eval_set_v1.jsonl`

### 2.4 评测脚本

| 文件 | 功能 |
|---|---|
| [evaluation/run_zero_shot_baseline.py](../evaluation/run_zero_shot_baseline.py) | 零样本基线评测，支持4-bit/FP16、在线/离线模式 |

管道验证：用Qwen2.5-0.5B-Instruct离线模式跑50条样本，全部成功。

---

## 三、待完成事项

### 3.1 7B模型零样本评测（阻塞项）

**状态**：评测脚本和数据集已就绪，但**7B模型无法下载**。

**原因**：当前网络无法连接 HuggingFace（无论开或关 VPN）：hf-mirror.com 也超时，Modelscope 也无法访问。

**需要用户操作**：
1. 尝试开启 VPN 后访问 HuggingFace
2. 或使用其他方式下载 Qwen2.5-7B-Instruct 到本地
3. 或使用已下载的7B模型路径

**下载后运行**：
```bash
python evaluation/run_zero_shot_baseline.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --eval_file data/processed/eval_set_v1.jsonl \
    --output_dir reports/baselines/qwen2.5-7b-instruct
```

### 3.2 轻量SFT对照（待7B模型可用后）

需要为每个候选模型执行相同的2k-5k样本QLoRA-SFT，对比SFT可塑性。

### 3.3 14B模型评测

14B模型在8GB GPU上仅能推理评测（4-bit），无法稳定训练。如需要，可作为纯推理对照。

---

## 四、当前交付物状态

| 交付物 | 状态 |
|---|---|
| 基座选型分析 | ✅ [docs/base_model_candidates.md](../docs/base_model_candidates.md) |
| 评测数据集 (550条) | ✅ data/processed/eval_set_v1.jsonl |
| 评测脚本（含离线模式） | ✅ evaluation/run_zero_shot_baseline.py |
| 统一配置 | ✅ configs/fixed_params.md |
| 评测管道验证 | ✅ 0.5B模型50条测试通过 |
| 零样本基线结果（7B） | ✅ 550条全量完成（0 failures） |
| 轻量SFT对照 | ⏳ 待步骤3数据准备后执行 |
| 资源消耗报告 | ✅ 5.2GB GPU / 49min / 550条 |
| 主线模型选定 | ✅ Qwen2.5-7B-Instruct |

---

## 五、踩坑记录

### 坑1：HuggingFace网络完全无法访问

**现象**：无论开/关VPN，huggingface.co 和 hf-mirror.com 均超时（WinError 10060/10054）。

**临时方案**：
- 评测脚本增加 `--offline` 模式，使用 `local_files_only=True`
- 0.5B模型已缓存，可验证管道
- 7B模型需用户手动下载

### 坑2：transformers即使有缓存仍尝试网络请求

**现象**：模型权重已在本地缓存，但加载时仍尝试请求 `generation_config.json`，导致超时等待。

**解决**：`AutoTokenizer.from_pretrained(..., local_files_only=True)` 和 `AutoModelForCausalLM.from_pretrained(..., local_files_only=True)`

**教训**：中国网络环境下做LLM项目，应优先考虑离线模式和本地模型管理。

---

## 六、下一步

进入**步骤3：设计数据体系与统一schema**。

---

## 七、7B全量评测补充结果（2026-08-04）

| 指标 | 值 |
|---|---|
| 评测样本 | 550 条（全量） |
| 成功率 | 100%（0 failures） |
| 模型加载时间 | 16.4s |
| 总生成时间 | 2963s（~49分钟） |
| 平均生成速度 | 5.39s/条 |
| GPU 显存占用 | 5.2GB / 8GB |
| 平均回复长度 | 222 字符 |

**按场景类型的回复长度分布**：

| 场景类型 | 样本数 | 平均长度 | 解读 |
|---|---|---|---|
| listen_only | 87 | **95 chars** | 最简洁，模型在倾听而非给建议 ✅ |
| multi_turn | 67 | 179 chars | 中等，关注前文信息 |
| safety_boundary | 58 | 217 chars | 安全场景适度展开 |
| ask_for_plan | 84 | 223 chars | 给计划但不过载 |
| common_stress | 186 | 242 chars | 标准共情+澄清 |
| reject_advice | 68 | **373 chars** | 最详细，需要重新调整策略 ✅ |

**关键发现**：
- `listen_only` 场景回复最短（95 chars），模型确实在"倾听"而非强行给建议
- `reject_advice` 场景回复最长（373 chars），模型在认真调整被拒绝的策略
- 14B模型评测跳过：8GB显存不足（4-bit权重~8GB，无剩余空间）
