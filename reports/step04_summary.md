# 步骤4总结：收集公开或授权数据

> 完成日期：2026-08-04

---

## 一、步骤任务回顾

根据纯模型训练版，步骤4包含7项任务：

1. 收集公开数据（同理心对话、情绪支持、压力和困扰表达等）
2. 对每个数据集核查许可、用途限制、修改权限、PII、训练许可
3. 删除与项目边界冲突的数据
4. 将数据转换为统一 schema
5. 非中文数据处理
6. 记录原始到处理数据的完整映射
7. 建议 SFT 数据中公开数据占 25%–40%

---

## 二、已完成内容

### 2.1 产出文档

| 文件 | 内容 |
|---|---|
| [docs/public_data_catalog.md](../docs/public_data_catalog.md) | 公开数据目录 + 许可审计（5个候选数据集） |
| [data_processing/convert_esconv.py](../data_processing/convert_esconv.py) | ESConv→统一schema转换脚本 |
| data/processed/public_data/esconv_sft_v1.jsonl | ESConv-LLM 910条转换后数据 |
| data/processed/public_data/adorable_eq_sft_v1.jsonl | Adorable EQ 170条转换后数据 |

### 2.2 已下载数据集

| 数据集 | 规模 | 语言 | 许可 | 状态 | 策略标签 |
|---|---|---|---|---|---|
| **ESConv-LLM** | 910 条对话 | 英文 | Apache-2.0 | ✅ 已转为统一schema | 8种策略标签 |
| **Adorable EQ Chat** | 170 条对话 | 中文 | CC-BY-4.0 | ✅ 已转为统一schema | 无（高EQ风格参考） |
| **ESConv (原始)** | 910 条 | 英文 | Apache-2.0 | ✅ 已下载（raw text格式） | — |

### 2.3 转换后数据统计

**ESConv-LLM (910条)**：

| 项目 | 值 |
|---|---|
| 平均轮数 | 18-29 轮/对话 |
| 策略标签 | ST-OPN(提问), ST-RFL(共情), ST-VAL(确认), ST-SUM(总结), ST-MIC(建议) |
| 情感分布 | anxiety(焦虑), depression(抑郁), loneliness(孤独), sadness(悲伤) |
| 待处理 | **需要翻译为中文**（英文数据集） |

**Adorable EQ (170条)**：

| 项目 | 值 |
|---|---|
| 对话类型 | 单轮高情商中文回应 |
| 严重度 | 统一标记为 SV-MLD（轻度） |
| 用途 | 中文自然度参考 + 风格多样性 |

### 2.4 未获取的数据集

以下中文数据集在 hf-mirror 镜像上不可用（可能已被删除、设为私有或镜像未同步）：

| 数据集 | 优先恢复 |
|---|---|
| MeChat/SmileChat (55K 中文) | ⭐⭐⭐ 最大中文情绪支持数据集 |
| CPsyCounD (3.1K 中文) | ⭐⭐⭐ 来自真实咨询报告 |
| ESCoT (1.7K 中文) | ⭐⭐⭐ 有策略推理标注 |
| SoulChat (258K 中文) | ⭐⭐ 可用于扩充 |

**替代方案**：
1. 尝试从原始 GitHub 仓库直接下载（不通过 HF）
2. 检查 ModelScope 是否有镜像
3. 使用百度网盘/阿里云盘等国内网盘（如作者提供）

---

## 三、踩坑记录

### 坑1：hf-mirror.com 环境变量设置方式

**现象**：在 Python 代码中用 `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'` 无效，仍报连接错误。

**解决**：必须在启动 Python 前通过 shell 导出：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python script.py
```

**教训**：`datasets` 库在 import 时就初始化了 HTTP 连接，Python 内设置环境变量太晚。

### 坑2：VPN对HTTPS的影响

**现象**：开VPN时 HF 报 ConnectionResetError(10054)，关VPN时 HF 报 ConnectTimeout(10060)。ModelScope 在关VPN时可用。

**结论**：当前网络环境下，hf-mirror.com（关VPN时可用）是唯一可用的HF数据源。

---

## 四、当前数据池状态

| 来源 | 目标占比 | 当前 | 缺口 |
|---|---|---|---|
| 公开数据 | 25%–40% (5k–12k) | 1,080 条(ESConv+Adorable) | ~4k–11k |
| 专家数据 | 20%–30% (4k–9k) | 0 | 全部待做(步骤5) |
| LLM合成 | 30%–50% (6k–15k) | 0 | 全部待做(步骤6) |

**结论**：公开数据端当前仅覆盖 MVP 目标的 ~5%。需要步骤5（专家数据）和步骤6（LLM合成数据）来补足。

---

## 五、下一步

进入**步骤5：构建专家编写与专家审校数据** — 这是纯文档/流程设计工作，不需要网络。
