"""
SFT-Stage1 训练脚本：领域行为对齐

用 QLoRA 在 final_split 数据上训练 Qwen2.5-7B-Instruct。
基于步骤8基线问题重点解决：
  1. 抢先建议 / 建议过载 / 策略时机（先倾听后建议）
  2. 共情具体化
  3. 风险识别

使用标准 Trainer + 自定义预处理（assistant掩码），避开trl兼容问题。

用法:
  python training/sft/run_sft_stage1.py --train_file data/processed/final_split/train.jsonl \
      --eval_file data/processed/final_split/dev.jsonl --output_dir checkpoints/sft_stage1
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "D:/anaconda3/envs/stress-mgmt/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"


def load_model():
    """加载4-bit QLoRA模型"""
    logger.info(f"Loading model: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def preprocess_example(example, tokenizer, max_len=2048):
    """预处理：chat_template编码，assistant部分计算loss（user部分掩码）"""
    try:
        messages = example["messages"]
        if not messages:
            return None

        # 分离：prompt（除最后一条assistant外的所有消息）+ completion（最后一条assistant）
        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        if len(messages) < 2:
            return None

        prompt_msgs = messages[:-1]
        completion = messages[-1]["content"]
        if not completion:
            return None

        # prompt用chat_template（含生成提示）
        prompt_enc = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
        # apply_chat_template 返回 dict 或 BatchEncoding
        if isinstance(prompt_enc, dict) or hasattr(prompt_enc, "keys"):
            prompt_ids = list(prompt_enc["input_ids"])
        else:
            prompt_ids = list(prompt_enc)

        # completion（截断确保不超max_len）
        completion_enc = tokenizer(
            completion, add_special_tokens=False, truncation=True,
            max_length=max_len,
        )
        completion_ids = list(completion_enc["input_ids"]) + [tokenizer.eos_token_id]

        # 截断到max_len
        remaining = max_len - len(prompt_ids)
        if remaining <= 0:
            return None
        completion_ids = completion_ids[:remaining]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        return {"input_ids": input_ids, "labels": labels}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/processed/final_split/train.jsonl")
    parser.add_argument("--eval_file", default="data/processed/final_split/dev.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/sft_stage1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=None, help="调试用")
    args = parser.parse_args()

    set_seed(42)

    # 加载模型
    model, tokenizer = load_model()

    # 加载数据
    logger.info(f"Loading train: {args.train_file}")
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train") if args.eval_file and os.path.exists(args.eval_file) else None
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds) if eval_ds else 0}")

    # 预处理（保证 input_ids 和 labels 等长、扁平list）
    def _process(ds):
        ds = ds.map(
            lambda x: preprocess_example(x, tokenizer, args.max_len),
            remove_columns=ds.column_names,
            num_proc=1,
        )
        # 过滤无效样本（None 或 无 input_ids 或 长度不一致）
        def _valid(x):
            if not x or "input_ids" not in x or "labels" not in x:
                return False
            if not isinstance(x["input_ids"], list) or not isinstance(x["labels"], list):
                return False
            if len(x["input_ids"]) != len(x["labels"]):
                return False
            return len(x["input_ids"]) > 20
        ds = ds.filter(_valid)
        return ds

    logger.info("Preprocessing train dataset...")
    train_ds = _process(train_ds)

    eval_ds_proc = None
    if eval_ds:
        logger.info("Preprocessing eval dataset...")
        eval_ds_proc = _process(eval_ds)

    logger.info(f"Preprocessed train: {len(train_ds)}")

    # 训练参数
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"sft_stage1_{datetime.now().strftime('%Y%m%d_%H%M')}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds_proc else "no",
        eval_steps=250,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        run_name=run_name,
    )

    def data_collator(features):
        """自定义collator：手动pad input_ids和labels"""
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(x) for x in input_ids)
        pad_id = tokenizer.pad_token_id

        batch_input = torch.full((len(features), max_len), pad_id, dtype=torch.long)
        batch_labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, (ids, labs) in enumerate(zip(input_ids, labels)):
            batch_input[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            batch_labels[i, :len(labs)] = torch.tensor(labs, dtype=torch.long)

        return {
            "input_ids": batch_input,
            "attention_mask": (batch_input != pad_id).long(),
            "labels": batch_labels,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds_proc,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("=" * 50)
    logger.info("Starting SFT-Stage1 training...")
    logger.info(f"  Epochs: {args.epochs}, LR: {args.lr}")
    logger.info(f"  Effective batch: {args.batch_size * args.grad_accum}")
    logger.info(f"  Max steps: {args.max_steps or 'auto'}")
    logger.info("=" * 50)

    t0 = time.time()
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed/60:.1f} min")

    # 保存
    save_path = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Adapter saved to {save_path}")


if __name__ == "__main__":
    main()
