# 网页手动下载数据集指南

> 日期：2026-08-05 | 用途：浏览器下载当前命令行无法获取的数据集

---

## 为什么需要手动下载

当前网络环境下，命令行（Python/curl）无法直连 HuggingFace，但浏览器可以打开。以下数据集在 hf-mirror 镜像上不存在，需要从浏览器下载。

---

## 数据集清单（按优先级排序）

### ⭐⭐⭐ 1. MeChat / SmileChat（55K 中文情绪支持对话）

**HuggingFace 页面**：https://huggingface.co/datasets/qiuhuachuan/MeChat

下载方式（网页端）：
1. 打开上面链接
2. 在 "Files and versions" 标签页找到 `data/` 目录下的 parquet 或 json 文件
3. 点击文件名右侧下载按钮（或打开后点 Download）
4. 文件通常为 `.parquet` 或 `.json`

**备选 GitHub**：https://github.com/qiuhuachuan/smile
- 仓库 `data/` 目录下有原始 jsonl 文件

**用途**：最大中文情绪支持数据集，可提供 ~40K 有效样本

---

### ⭐⭐⭐ 2. CPsyCounD（3.1K 中文心理咨询多轮对话）

**HuggingFace 页面**：https://huggingface.co/datasets/qiuhuachuan/CPsyCounD

下载方式：
1. 打开链接 → Files 标签
2. 下载 `dataset.json` 或 parquet 文件

**备选 GitHub**：https://github.com/CAS-SIAT-XinHai/CPsyCounD

**用途**：来自真实咨询报告重建，质量高

---

### ⭐⭐⭐ 3. ESCoT（1.7K 中文情绪支持 + CoT策略推理）

**GitHub 页面**：https://github.com/TeigenZhang/ESCoT

下载方式：
1. 打开 GitHub 仓库
2. 点绿色 "Code" 按钮 → "Download ZIP"
3. 解压后数据在 `data/` 目录

**用途**：带策略推理标注，直接映射到我们的策略标签

---

### ⭐⭐ 4. SoulChat（258K 中文共情对话）

**HuggingFace 页面**：https://huggingface.co/datasets/scutcyr/SoulChat

下载方式：
1. 打开链接 → Files 标签
2. 下载 jsonl 文件（可能较大，几GB）

**注意**：该数据集为LLM合成，质量需抽检

---

### ⭐⭐ 5. ESConv-SRA（英文，长程情绪支持续写）

**HuggingFace 页面**：https://huggingface.co/datasets/navidmdn/ESConv-SRA

下载方式：同 MeChat

**用途**：需翻译，但可补充长程对话场景

---

## 下载后处理流程

下载好的文件放到项目 `data/raw/` 目录，我会编写对应的转换脚本：

```
data/raw/
├── mechat/          # MeChat 原始文件
│   └── train.parquet (或 .json)
├── cpsycd/          # CPsyCounD 原始文件
├── escot/           # ESCoT 原始文件
└── soulchat/        # SoulChat 原始文件
```

放好后告诉我文件位置，我会：
1. 审计许可（在 [public_data_catalog.md](public_data_catalog.md) 登记）
2. 编写格式转换脚本
3. 转换到统一 schema
4. 合并进清洗管道

---

## 浏览器 vs 命令行下载区别

| 方式 | 能否用 | 原因 |
|---|---|---|
| 命令行 (Python) | ❌ | VPN 下所有 HTTP 库报 ConnectionReset |
| hf-mirror | 部分 ✅ | 但 MeChat 等中文数据集未同步 |
| **浏览器 (网页)** | ✅ | 用系统网络栈，能通过 VPN |

> 提示：如果浏览器下载到一半断连，可以点 Resume 续传。大文件建议用浏览器下载而非命令行。
