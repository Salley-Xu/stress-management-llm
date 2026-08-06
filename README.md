# 压力管理适配大模型后训练项目

> 中文文本对话模型的后训练，专注日常压力管理、情绪支持、问题拆解和行动规划。

## 项目结构

```
后训练/
├── data/                   # 数据目录（训练数据不推送，脚本可重生成）
│   ├── raw/                #   原始数据
│   ├── processed/          #   清洗后数据
│   └── external/           #   外部数据
├── data_processing/        # 数据处理脚本
│   ├── convert_smilechat.py #   SmileChat 转换
│   ├── convert_escot.py    #   ESCoT 转换
│   ├── generate_synthetic.py # DeepSeek 合成数据
│   ├── generate_expert_data.py # 专家数据 + Rubric筛选
│   ├── clean_data.py       #   清洗去重(LSH)
│   ├── split_data.py       #   分层切分+泄漏审计
│   └── evaluate_training_data.py # 100分制质量评估
├── training/               # 训练脚本
│   ├── sft/
│   │   ├── run_sft_stage1.py # SFT-Stage1 领域对齐
│   │   └── smoke_test.py     # QLoRA Smoke Test
│   ├── preference/         #   偏好优化
│   └── rl/                 #   强化学习
├── evaluation/             # 评测脚本
│   ├── run_zero_shot_baseline.py # 零样本基线
│   ├── eval_sft.py         #   SFT后评测
│   └── deep_error_analysis.py  # LLM错误分析
├── serving/                # 推理服务
│   └── serve.py            #   vLLM/SGLang/Transformers 服务
├── configs/                # 配置文件
│   ├── default.yaml        #   默认训练/评测/服务配置
│   ├── judge_prompts.md    #   LLM-as-a-Judge prompt
│   └── naming_convention.md #  命名与版本规范
├── docs/                   # 项目文档（19份）
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
# SFT-Stage1：领域行为对齐
python training/sft/run_sft_stage1.py \
    --train_file data/processed/final_split/train.jsonl \
    --output_dir checkpoints/sft_stage1 \
    --epochs 1 --batch_size 1 --grad_accum 8
```

> 注：8GB GPU 上 7B QLoRA 单样本约18s，建议用高质量子集控制训练时长。

### 5. 评测

```bash
# 零样本基线
python evaluation/run_zero_shot_baseline.py \
    --model <model_path> --eval_file data/processed/eval_set_v1.jsonl \
    --output_dir reports/baselines/xxx

# SFT后评测（加载adapter）
python evaluation/eval_sft.py \
    --adapter checkpoints/sft_stage1/final_adapter \
    --output_dir reports/baselines/sft_stage1

# LLM错误分析
python evaluation/deep_error_analysis.py --num_samples 550
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
