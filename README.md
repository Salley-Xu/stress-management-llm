# 压力管理适配大模型后训练项目

> 中文文本对话模型的后训练，专注日常压力管理、情绪支持、问题拆解和行动规划。

## 项目结构

```
后训练/
├── data/                   # 数据目录
│   ├── raw/                #   原始数据
│   ├── processed/          #   清洗后数据
│   └── external/           #   外部数据
├── training/               # 训练脚本
│   ├── run_sft.py          #   SFT 训练入口
│   └── smoke_test.py       #   QLoRA Smoke Test
├── evaluation/             # 评测脚本
│   └── eval_basic.py       #   基础自动评测
├── serving/                # 推理服务
│   └── serve.py            #   vLLM/SGLang/Transformers 服务
├── configs/                # 配置文件
│   ├── default.yaml        #   默认训练/评测/服务配置
│   └── naming_convention.md #  命名与版本规范
├── scripts/                # 工具脚本
│   └── verify_env.py       #   环境验证
├── reports/                # 报告输出
├── requirements.txt        # Python 依赖（版本锁定）
├── pyproject.toml          # 项目元信息
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 验证环境
python scripts/verify_env.py
```

### 2. 配置

```bash
# 复制环境变量
cp .env.example .env
# 编辑 .env 填入 API keys

# 检查默认配置
python training/run_sft.py --config configs/default.yaml --dry_run
```

### 3. Smoke Test

```bash
# 验证硬件可以完成 QLoRA 训练
python training/smoke_test.py
```

### 4. SFT 训练

```bash
# 使用默认配置
python training/run_sft.py --config configs/default.yaml

# 自定义参数
python training/run_sft.py --config configs/default.yaml \
    --model.base_model_name Qwen/Qwen2.5-7B-Instruct \
    --qlora.lora_r 64 \
    --training.num_train_epochs 3 \
    --tracking.run_name my-experiment
```

### 5. 评测

```bash
python evaluation/eval_basic.py \
    --model_path checkpoints/sm-sft-xxx/final_adapter \
    --base_model_name Qwen/Qwen2.5-7B-Instruct \
    --eval_file data/processed/eval/test.jsonl \
    --output_dir reports/my-eval
```

### 6. 推理服务

```bash
# vLLM API 服务
python serving/serve.py --engine vllm --model_path checkpoints/xxx --port 8000

# 交互式对话测试
python serving/serve.py --engine transformers --model_path checkpoints/xxx --interactive
```

## 实验追踪

- 使用 W&B (Weights & Biases) 记录实验
- 每项实验需记录：模型版本、数据版本、随机种子、训练参数、评测结果
- 详细命名规范见 [configs/naming_convention.md](configs/naming_convention.md)

## 安全声明

- 本模型仅用于日常压力管理支持，不用于心理疾病诊断、治疗决策或药物建议
- 高风险场景需配置独立风险分类器和人工升级链路
- 热线和机构信息应作为外部配置，禁止模型自由生成

## 技术栈

- **基座模型**: Qwen3.5-9B
- **训练**: QLoRA (4-bit) + SFT + DPO/ORPO
- **推理**: vLLM / SGLang
- **追踪**: W&B / MLflow
- **框架**: Transformers + PEFT + TRL + Accelerate
