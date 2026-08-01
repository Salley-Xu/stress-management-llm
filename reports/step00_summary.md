# 步骤0总结：建立项目管理与复现环境

> 完成日期：2026-08-01

---

## 一、步骤任务回顾

根据《压力管理后训练项目_执行步骤.md》，步骤0包含6项任务：

1. 建立代码仓库，划分 `data/`、`training/`、`evaluation/`、`serving/`、`configs/`、`reports/`
2. 固定 Python、CUDA、PyTorch、Transformers、PEFT、TRL、vLLM/SGLang 等版本
3. 接入实验追踪工具，记录模型版本、数据版本、随机种子、训练参数和评测结果
4. 建立数据与模型版本命名规则
5. 建立配置驱动的训练、评测和推理脚本
6. 在目标 GPU 上完成一次最小化 QLoRA smoke test

**通过条件**：
- 同一配置重复运行可得到一致结果
- 能在目标硬件上完成数据加载、前向、反向、保存和推理

---

## 二、已完成内容

### 2.1 项目目录结构（7个目录 + 22个文件）

```
后训练/
├── .env.example              # 环境变量模板
├── .gitignore                # Git忽略规则
├── README.md                 # 项目文档
├── pyproject.toml            # 项目构建配置
├── requirements.txt          # 依赖锁定（37个包）
├── configs/
│   ├── default.yaml          # 统一配置（7大类：模型/QLoRA/训练/数据/追踪/评测/安全）
│   └── naming_convention.md  # 6类命名规范（模型/数据/实验标签/路径/分支/种子）
├── training/
│   ├── __init__.py
│   ├── smoke_test.py         # QLoRA Smoke Test（6步验证流程）
│   └── run_sft.py            # SFT训练入口（支持CLI覆盖配置）
├── evaluation/
│   ├── __init__.py
│   └── eval_basic.py         # 基础评测（ROUGE/BERTScore）
├── serving/
│   ├── __init__.py
│   └── serve.py              # 推理服务（vLLM/SGLang/交互式）
├── scripts/
│   └── verify_env.py         # 环境依赖验证
├── data/{raw,processed,external}/  # 数据目录（含.gitkeep）
└── reports/                  # 报告目录
```

### 2.2 Conda环境

| 项目 | 值 |
|---|---|
| 环境名称 | `stress-mgmt` |
| 位置 | `D:\anaconda3\envs\stress-mgmt` |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| CUDA Runtime | 13.0 |
| 关键包 | transformers 5.14.1, peft 0.20.0, bitsandbytes 0.50.0, trl 1.9.2, accelerate 1.14.0, datasets 5.0.1 |

激活命令：
```bash
source /d/anaconda3/etc/profile.d/conda.sh && conda activate stress-mgmt
```

### 2.3 硬件验证结果

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| 显存 | 8.0 GB |
| 驱动版本 | 591.86 |
| CUDA 支持 | CUDA 13.1 (驱动) / CUDA 13.0 (PyTorch) |
| 计算能力 | sm_120 (Blackwell) |

### 2.4 Smoke Test 结果

| 步骤 | 内容 | 耗时 | 结果 |
|---|---|---|---|
| 1 | 环境检测 | <1s | ✅ |
| 2 | Tokenizer加载 (Qwen2.5-0.5B) | ~16s | ✅ |
| 3 | 4-bit QLoRA模型加载 | ~4min | ✅ (494M params, 0.7GB显存) |
| 4 | LoRA配置 | <1s | ✅ (trainable 1.91%) |
| 5 | 训练10步 | 3.7s | ✅ (loss 3.22→0.82) |
| 6 | Adapter保存+推理 | ~3s | ✅ (中文输出正常) |

---

## 三、踩过的坑与解决方案

### 坑1：conda下载超时（默认源在国内网络不稳定）

**现象**：`conda create` 和 `conda install` 频繁出现 `ConnectionResetError(10054)` 或 `ReadTimeoutError`。

**排查**：
- 起初以为需要国内镜像，配置了清华tuna镜像
- 镜像同样报连接重置错误

**解决**：用户反馈VPN开启导致连接问题，关闭VPN后默认源正常。

**教训**：网络问题时优先检查代理/VPN状态，而非直接换源。

---

### 坑2：conda安装PyTorch时路径太长导致Windows报错

**现象**：
```
InvalidArchiveError: [Errno 2] No such file or directory:
'...\\pytorch-2.12.0-gpu_cuda130_py310h1c49258_300\\info\\test\\test\\compiled_autograd_skips\\TestNestedTensorSubclassCPU...'
```

**原因**：conda包的测试文件路径嵌套太深，超过Windows 260字符路径限制。

**解决**：改用 `pip install torch --index-url https://download.pytorch.org/whl/cu130` 安装，pip的wheel包路径更简洁。

**教训**：Windows下conda安装大型包容易触发长路径问题，PyTorch优先用pip。

---

### 坑3：RTX 5070 (Blackwell sm_120) 与 CUDA 12.6 版 PyTorch 不兼容

**现象**：
```
NVIDIA GeForce RTX 5070 Laptop GPU with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 ... sm_90.
```

**原因**：RTX 5070 是 Blackwell 架构 (CC 12.0)，PyTorch 2.13+cu126 只编译到 sm_90。需要 cu129、cu130 或 cu132 版本。

**解决**：卸载 `torch-2.13.0+cu126`，重装 `torch-2.13.0+cu130`。

```bash
pip install torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

**教训**：新GPU架构（Blackwell 2025年后）需要匹配的CUDA编译版本。PyTorch官方提供了 cu129/cu130/cu132 多种选择，需要根据驱动版本选择。

---

### 坑4：conda环境缺少SSL证书文件

**现象**：
```
FileNotFoundError: [Errno 2] No such file or directory
...
ctx = ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])
```

**原因**：conda创建环境时设置了 `SSL_CERT_FILE=D:\anaconda3\envs\stress-mgmt/ssl/cacert.pem`，但最小化环境没有安装 `ca-certificates` 包，该文件不存在。

**解决**：
```bash
mkdir -p /d/anaconda3/envs/stress-mgmt/ssl
cp /d/anaconda3/envs/stress-mgmt/lib/site-packages/certifi/cacert.pem \
   /d/anaconda3/envs/stress-mgmt/ssl/cacert.pem
```

**教训**：`conda create -n env python=3.10` 创建的最小环境不包含SSL证书。如果不需要conda管理SSL，可 `unset SSL_CERT_FILE` 让Python使用certifi。

---

### 坑5：vLLM/SGLang 无法在Windows安装

**现象**：
```
ERROR: [Errno 2] No such file or directory: '...\\vllm\\model_executor\\layers\\mamba\\ops\\configs\\selective_state_update\\headdim=64...'
```

**原因**：vLLM内部文件名/路径嵌套极深，远超Windows 260字符限制。且vLLM/SGLang本身主要为Linux设计。

**当前方案**：跳过vLLM/SGLang。本地测试使用 `serve.py --engine transformers --interactive`。生产环境在Linux服务器部署。

---

### 坑6：终端编码问题导致脚本crash

**现象**：
```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705' in position 2
```

**原因**：Windows终端使用GBK编码，脚本中的emoji（✅❌⚠️）无法输出。

**解决**：将 `verify_env.py` 中所有emoji替换为ASCII标记 `[OK]`、`[FAIL]`、`[WARN]`。

**教训**：Windows下Python脚本避免使用emoji和特殊Unicode字符，用纯ASCII状态标记替代。

---

## 四、当前局限与后续注意事项

| 局限 | 影响 | 建议 |
|---|---|---|
| Windows环境 | vLLM/SGLang无法安装 | 训练在Windows完成，服务部署迁移到Linux |
| 8GB VRAM | 无法全精度训练7B+模型 | QLoRA 4-bit可用；生产建议≥24GB显卡 |
| 无Flash Attention | 训练/推理略慢 | Windows不支持，Linux部署时安装 |
| HuggingFace无缓存 | 首次下载模型耗时较长 | 提前下载常用模型到本地缓存 |
| 未配置W&B Token | 实验追踪未实际激活 | 在`.env`中填写`WANDB_API_KEY` |

---

## 五、下一步

进入**步骤1：定义产品目标、场景和安全边界**，产出：
- 产品需求文档 PRD
- 能力边界说明
- 风险分级表 (L0-L3)
- Safety Spec
- 场景与任务分类表
