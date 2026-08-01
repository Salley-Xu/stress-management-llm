"""
QLoRA SFT 训练入口脚本

用法:
    # 使用默认配置
    python training/run_sft.py --config configs/default.yaml

    # 覆盖部分参数
    python training/run_sft.py --config configs/default.yaml \
        --model.base_model_name Qwen/Qwen2.5-7B-Instruct \
        --qlora.lora_r 64 \
        --training.num_train_epochs 3 \
        --tracking.run_name sm-sft-experiment-01

    # 仅 dry run 验证配置
    python training/run_sft.py --config configs/default.yaml --dry_run
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml
from omegaconf import OmegaConf
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# 加载环境变量
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> OmegaConf:
    """加载 YAML 配置，支持变量替换"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # 支持 ${env:VAR_NAME} 环境变量替换
    cfg = OmegaConf.create(raw)
    return cfg


def setup_wandb(cfg, model_version: str) -> Optional[str]:
    """初始化 W&B 实验追踪"""
    if cfg.tracking.tool != "wandb":
        return None

    run_name = cfg.tracking.run_name or f"sm-sft-{model_version}-{datetime.now():%Y%m%d-%H%M%S}"
    wandb.init(
        project=cfg.tracking.wandb_project,
        entity=cfg.tracking.wandb_entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.tracking.tags or [],
    )
    logger.info(f"  W&B run: {run_name}")
    return run_name


def load_model_and_tokenizer(cfg):
    """加载基座模型和 tokenizer（含 QLoRA 量化配置）"""
    logger.info("=" * 60)
    logger.info(f"加载模型: {cfg.model.base_model_name}")
    logger.info("=" * 60)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model_name,
        trust_remote_code=cfg.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"  Tokenizer vocab_size={tokenizer.vocab_size}")

    # ---- QLoRA 量化配置 ----
    if cfg.qlora.enabled:
        compute_dtype = getattr(torch, cfg.model.torch_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(cfg.qlora.bits == 4),
            load_in_8bit=(cfg.qlora.bits == 8),
            bnb_4bit_quant_type=cfg.qlora.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.qlora.double_quant,
            bnb_4bit_quant_storage=compute_dtype,
        )
        logger.info(f"  QLoRA: {cfg.qlora.bits}-bit, quant_type={cfg.qlora.quant_type}")
    else:
        bnb_config = None

    # ---- 模型加载 ----
    attn_impl = cfg.model.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model_name,
            quantization_config=bnb_config,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        logger.warning(f"  {attn_impl} 加载失败，尝试 sdpa: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model_name,
            quantization_config=bnb_config,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
            attn_implementation="sdpa",
        )

    if cfg.qlora.enabled:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded. total_params={total_params:,}")

    # ---- LoRA 配置 ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.qlora.lora_r,
        lora_alpha=cfg.qlora.lora_alpha,
        lora_dropout=cfg.qlora.lora_dropout,
        target_modules=list(cfg.qlora.target_modules),
        modules_to_save=list(cfg.qlora.modules_to_save),
        bias="none",
    )

    return model, tokenizer, peft_config


def load_sft_data(cfg, tokenizer):
    """加载并预处理 SFT 数据"""
    logger.info("=" * 60)
    logger.info("加载训练数据")
    logger.info("=" * 60)

    train_file = cfg.data.train_file
    eval_file = cfg.data.eval_file

    if not os.path.exists(train_file):
        logger.warning(f"  训练数据文件不存在: {train_file}")
        logger.warning("  将使用 dummy 数据以验证流程")
        return _create_dummy_sft_data(tokenizer, 100), None

    train_dataset = load_dataset("json", data_files=train_file, split="train")
    logger.info(f"  训练样本数: {len(train_dataset)}")

    eval_dataset = None
    if os.path.exists(eval_file):
        eval_dataset = load_dataset("json", data_files=eval_file, split="train")
        logger.info(f"  验证样本数: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def _create_dummy_sft_data(tokenizer, num_samples: int = 100) -> Dataset:
    """创建虚拟 SFT 数据用于流程验证"""
    samples = []
    for i in range(num_samples):
        samples.append({
            "messages": [
                {"role": "system", "content": "你是一个提供日常压力管理支持的助手。"},
                {"role": "user", "content": f"示例用户消息 {i}: 我最近压力很大"},
                {"role": "assistant", "content": f"示例回复 {i}: 我理解你的感受，可以多聊聊具体情况吗？"},
            ]
        })
    return Dataset.from_list(samples)


def formatting_func(example, tokenizer):
    """将 messages 格式化为训练文本"""
    if "messages" in example:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return text
    elif "text" in example:
        return example["text"]
    return ""


def run_sft(cfg, dry_run: bool = False):
    """执行 QLoRA SFT 训练"""
    # ---- 版本信息 ----
    model_name_short = Path(cfg.model.base_model_name).name
    model_version = f"{model_name_short}-r{cfg.qlora.lora_r}-lr{cfg.training.learning_rate}"

    logger.info("=" * 60)
    logger.info(f"QLoRA SFT 训练启动")
    logger.info(f"  模型版本: {model_version}")
    logger.info(f"  Dry run: {dry_run}")
    logger.info(f"  随机种子: {cfg.project.seed}")
    logger.info("=" * 60)

    # ---- 设置随机种子 ----
    set_seed(cfg.project.seed)

    # ---- 初始化追踪 ----
    run_name = setup_wandb(cfg, model_version)

    # ---- 加载模型 ----
    model, tokenizer, peft_config = load_model_and_tokenizer(cfg)

    # ---- 加载数据 ----
    train_dataset, eval_dataset = load_sft_data(cfg, tokenizer)

    if dry_run:
        logger.info("  [Dry Run] 跳过实际训练，配置验证通过。")

        # 打印关键配置摘要
        logger.info("=" * 60)
        logger.info("配置摘要")
        logger.info("=" * 60)
        logger.info(OmegaConf.to_yaml(cfg))
        return

    # ---- 构建训练参数 ----
    output_dir = os.path.join(
        cfg.training.output_dir, f"sm-sft-{model_version}-{datetime.now():%Y%m%d}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        bf16=cfg.training.bf16,
        tf32=cfg.training.tf32,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=cfg.training.remove_unused_columns,
        report_to=cfg.tracking.tool,
        run_name=run_name,
        seed=cfg.project.seed,
    )

    # ---- 使用 SFTTrainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=lambda x: formatting_func(x, tokenizer),
        max_seq_length=cfg.data.max_seq_length,
    )

    # ---- 开始训练 ----
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60)

    try:
        train_result = trainer.train()
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

    # ---- 保存最终模型 ----
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    logger.info(f"  Adapter saved to {final_adapter_path}")

    # ---- 保存训练指标 ----
    metrics = {
        "model_version": model_version,
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_result.metrics.get("train_runtime", 0),
        "global_step": train_result.global_step,
        "output_dir": output_dir,
        "adapter_path": final_adapter_path,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"  Metrics saved to {metrics_path}")

    # ---- 清理 ----
    if cfg.tracking.tool == "wandb":
        wandb.finish()

    logger.info("=" * 60)
    logger.info(f"训练完成！")
    logger.info(f"  Adapter: {final_adapter_path}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info("=" * 60)

    return trainer


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅验证配置，不执行训练")
    # 支持命令行覆盖配置
    parser.add_argument("--model.base_model_name", type=str)
    parser.add_argument("--qlora.lora_r", type=int)
    parser.add_argument("--training.num_train_epochs", type=float)
    parser.add_argument("--training.learning_rate", type=float)
    parser.add_argument("--tracking.run_name", type=str)
    parser.add_argument("--data.train_file", type=str)
    parser.add_argument("--data.eval_file", type=str)

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行覆盖
    cli_overrides = {}
    for key in [
        "model.base_model_name", "qlora.lora_r",
        "training.num_train_epochs", "training.learning_rate",
        "tracking.run_name", "data.train_file", "data.eval_file",
    ]:
        val = getattr(args, key.replace(".", "__"), None) or getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val

    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))

    run_sft(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
