# 实验命名与版本规范

## 1. 模型版本命名

```
{project}-{step}-{base_model}-{date}-{version_tag}
```

示例：
```
sm-sft-qwen3.5-9b-20240801-v1
sm-dpo-qwen3.5-9b-20240805-beta01
sm-safety-qwen3.5-9b-20240810-v2
```

| 字段 | 说明 | 示例值 |
|---|---|---|
| project | 项目缩写 | `sm` (stress-management) |
| step | 训练阶段 | `sft`, `dpo`, `safety`, `rlhf` |
| base_model | 基座模型 | `qwen3.5-9b`, `qwen3.5-9b-base` |
| date | 训练日期 | `20240801` |
| version_tag | 版本标签 | `v1`, `v2`, `beta01` |

## 2. 数据版本命名

```
{project}-{data_type}-{size}-{date}-{version}
```

示例：
```
sm-sft-15k-20240801-v1
sm-pref-8k-20240805-v1
sm-safety-6k-20240810-v1
```

| 字段 | 说明 |
|---|---|
| data_type | 数据类型: `sft`, `pref`(偏好), `safety`, `eval` |
| size | 数据量: `5k`, `15k`, `35k` 等 |
| date | 数据冻结日期 |
| version | 标注/清洗版本号 |

## 3. 实验追踪标签

每次训练/评测运行必须记录的标签：

| 标签 | 说明 | 必填 |
|---|---|---|
| `model_version` | 模型版本名称 | ✅ |
| `data_version` | 数据版本名称 | ✅ |
| `random_seed` | 随机种子 | ✅ |
| `base_model` | 基座模型名称 | ✅ |
| `lora_rank` | LoRA rank | ✅ |
| `learning_rate` | 学习率 | ✅ |
| `num_epochs` | 训练轮数 | ✅ |
| `batch_size` | 批次大小 | ✅ |
| `context_length` | 上下文长度 | ✅ |
| `quantization` | 量化方式 | ✅ |
| `gpu_type` | GPU 型号 | ✅ |
| `cuda_version` | CUDA 版本 | ✅ |
| `train_data_size` | 训练数据量 | ✅ |
| `eval_data_version` | 评测数据版本 | ✅ |

## 4. 文件路径规范

```
{root}/
├── data/
│   ├── raw/{data_version}/       # 原始数据
│   └── processed/{data_version}/ # 清洗后数据
├── checkpoints/{model_version}/  # 模型检查点
├── logs/{model_version}/         # 训练日志
├── reports/{model_version}/      # 评测报告
└── configs/{model_version}.yaml  # 训练配置快照
```

## 5. Git 分支规范

| 分支类型 | 命名 | 说明 |
|---|---|---|
| 主分支 | `main` | 稳定版本 |
| 开发分支 | `dev` | 集成分支 |
| 实验分支 | `exp/{step}/{description}` | 如 `exp/sft/lora-rank-64` |
| 数据分支 | `data/{data_type}/{description}` | 如 `data/sft/add-expert-samples` |
| 发布分支 | `release/{version}` | 如 `release/v0.1.0` |

## 6. 随机种子约定

- 默认种子: `42`
- 如需多次运行取平均，使用种子: `42, 123, 456, 789, 1024`
- 所有实验必须在配置文件中明确记录种子值
