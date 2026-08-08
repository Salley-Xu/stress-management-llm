"""
SFT-Stage2 训练脚本：多轮与安全专项训练

以 SFT-Stage1 final_adapter 为起点，继续训练：
  数据 = 多轮训练集V2(M1-M6) + 安全专项训练集V2(S1-S6, 上采样) + Stage1 replay
目标：解决 SFT-Stage1 遗留问题（风险漏检↑、多轮状态丢失、回复过短）

为什么用手动训练循环而非 Trainer：
  本机8GB显存上 Trainer 训练极慢（gradient checkpointing 在本机不降显存
  反而重算导致每步>8min），而手动循环（forward+backward+step）实测稳定5.7s/步。

关键设置（步骤10要求）：
  - 较低学习率 1e-4（防止单轮能力和自然度退化）
  - 混入 Stage1 replay 数据
  - 安全专项上采样（数据侧已处理）
  - max_len=448（8GB显存上限，>512 tokens的激活会超物理显存）

用法:
  python training/sft/run_sft_stage2.py --train_file data/processed/sft2_data/sft2_pool_*.jsonl \
      --output_dir checkpoints/sft_stage2
"""

import json
import os
import time
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from peft import PeftModel, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "D:/anaconda3/envs/stress-mgmt/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
STAGE1_ADAPTER = "checkpoints/sft_stage1/final_adapter"


def load_model(adapter_path=STAGE1_ADAPTER):
    """加载 base(4bit) + 指定 adapter（默认Stage1），继续训练"""
    logger.info(f"Loading base model: {MODEL_PATH}")
    logger.info(f"Loading adapter: {adapter_path}")

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
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # is_trainable=True 关键：否则 LoRA 权重 requires_grad=False，训练无效
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    # gradient checkpointing 必须保持启用：本机8GB显存下它省显存（实测关闭后激活全保存
    # 超物理显存触发共享内存交换，训练极慢；启用时实测5.7s/步稳定）。
    # 显式指定 use_reentrant=False（PyTorch 2.13 推荐，避免默认reentrant告警）
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def find_resume_checkpoint(output_dir: Path):
    """查找可恢复的最新 checkpoint（{output_dir}/checkpoints/step_N/）"""
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    steps = [d for d in ckpt_dir.glob("step_*") if d.is_dir() and (d / "train_state.json").exists()]
    if not steps:
        return None
    return max(steps, key=lambda d: int(d.name.split("_")[1]))


def save_checkpoint(output_dir: Path, model, opt, scheduler, global_step, sample_idx):
    """保存断点：adapter权重 + 优化器 + scheduler + 训练位置"""
    ckpt_path = output_dir / "checkpoints" / f"step_{global_step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    torch.save(opt.state_dict(), str(ckpt_path / "optimizer.pt"))
    torch.save(scheduler.state_dict(), str(ckpt_path / "scheduler.pt"))
    with open(ckpt_path / "train_state.json", "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step, "sample_idx": sample_idx}, f)
    logger.info(f"  [save] checkpoint -> {ckpt_path}")


def preprocess_example(example, tokenizer, max_len=320):
    """预处理：chat_template编码，只对最后一条assistant算loss
    多轮数据同样适用——prompt=历史轮次，completion=最后一条assistant回复。

    关键：8GB显存上7B QLoRA训练序列>448 tokens时个别样本会恰好超物理显存
    触发共享内存交换导致灾难性卡死（实测全量448卡在特定样本）。因此：
      - max_len默认320（实测最稳5.3s/步，保留最近2轮对话）
      - 只保留最近 max_history_msgs 条历史消息，超长prompt从头部截断
    """
    try:
        messages = example["messages"]
        if not messages:
            return None

        if messages[-1]["role"] != "assistant":
            messages = messages[:-1]
        if len(messages) < 2:
            return None

        prompt_msgs = messages[:-1]
        completion = messages[-1]["content"]
        if not completion:
            return None

        # 多轮：只保留最近若干条历史（最多3轮对话），避免prompt过长
        max_history_msgs = 6
        if len(prompt_msgs) > max_history_msgs:
            prompt_msgs = prompt_msgs[-max_history_msgs:]

        prompt_enc = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_enc, dict) or hasattr(prompt_enc, "keys"):
            prompt_ids = list(prompt_enc["input_ids"])
        else:
            prompt_ids = list(prompt_enc)

        # 若prompt仍超长，token级从头部截断（保留最近的轮次）
        reserve = max(64, max_len // 8)
        if len(prompt_ids) > max_len - reserve:
            prompt_ids = prompt_ids[-(max_len - reserve):]

        completion_enc = tokenizer(
            completion, add_special_tokens=False, truncation=True, max_length=max_len,
        )
        completion_ids = list(completion_enc["input_ids"]) + [tokenizer.eos_token_id]

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
    parser.add_argument("--train_file", default=None, help="sft2_pool jsonl")
    parser.add_argument("--output_dir", default="checkpoints/sft_stage2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=320)
    parser.add_argument("--max_steps", type=int, default=None, help="调试用")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50, help="每N个优化器step保存断点（支持续训）")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从 output_dir/checkpoints 最新断点续训（不传则自动检测）")
    args = parser.parse_args()

    # 自动选择最新的 sft2_pool
    if not args.train_file:
        pools = sorted(Path("data/processed/sft2_data").glob("sft2_pool_*.jsonl"))
        if not pools:
            logger.error("No sft2_pool_*.jsonl found. Run build_sft2_pool.py first.")
            return
        args.train_file = str(pools[-1])

    set_seed(42)
    random.seed(42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 断点检测
    resume_ckpt = find_resume_checkpoint(output_dir)
    if args.resume and not resume_ckpt:
        logger.warning("--resume 指定但未找到 checkpoint，从头训练")
    if resume_ckpt and not args.resume:
        logger.info(f"检测到断点 {resume_ckpt}，加 --resume 可续训；本次从头训练")
    adapter_path = resume_ckpt if (resume_ckpt and args.resume) else STAGE1_ADAPTER

    model, tokenizer = load_model(adapter_path)

    logger.info(f"Loading train: {args.train_file}")
    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    logger.info(f"Train: {len(train_ds)}")

    # 预处理
    train_examples = []
    for ex in train_ds:
        p = preprocess_example(ex, tokenizer, args.max_len)
        if p:
            train_examples.append(p)
    logger.info(f"Preprocessed train: {len(train_examples)}")

    # 固定seed shuffle（恢复时顺序一致，sample_idx才有效）
    random.shuffle(train_examples)

    # 优化器 + scheduler（手动实现梯度累积）
    opt = AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_examples) + effective_batch - 1) // effective_batch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    # 恢复优化器/scheduler/位置
    global_step = 0
    resume_idx = 0
    if resume_ckpt and args.resume:
        opt.load_state_dict(torch.load(resume_ckpt / "optimizer.pt", map_location="cpu"))
        scheduler.load_state_dict(torch.load(resume_ckpt / "scheduler.pt", map_location="cpu"))
        with open(resume_ckpt / "train_state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        global_step = state["global_step"]
        resume_idx = state["sample_idx"]
        logger.info(f"  [resume] from {resume_ckpt}: global_step={global_step}, sample_idx={resume_idx}")

    logger.info("=" * 50)
    logger.info("Starting SFT-Stage2 training (manual loop)...")
    logger.info(f"  Epochs: {args.epochs}, LR: {args.lr}, max_len: {args.max_len}")
    logger.info(f"  Effective batch: {effective_batch}, total_steps: {total_steps}")
    logger.info(f"  Data: {args.train_file}")
    logger.info(f"  Save checkpoint every {args.save_steps} steps (resume-capable)")
    logger.info("=" * 50)

    t0 = time.time()
    accum_count = 0
    loss_accum = 0.0
    model.train()

    def run_step(sample):
        nonlocal accum_count, loss_accum, global_step
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).cuda()
        labels = torch.tensor([sample["labels"]], dtype=torch.long).cuda()
        attn = (input_ids != tokenizer.pad_token_id).long()
        t_sample = time.time()
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t_sample
        if dt > 25:
            # 慢样本：清梯度跳过，避免该样本+后续样本拖成灾难（此前累积梯度作废，可接受）
            logger.warning(f"  [skip] 慢样本 len={len(input_ids[0])} took {dt:.1f}s，跳过其梯度")
            opt.zero_grad()
            return
        loss_accum += out.loss.item()
        accum_count += 1
        if accum_count >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            global_step += 1
            accum_count = 0
            if global_step % args.logging_steps == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    f"  step {global_step}/{total_steps} | loss {loss_accum/args.logging_steps:.4f} "
                    f"| lr {lr_now:.2e} | {elapsed:.0f}s"
                )
                loss_accum = 0.0

    last_saved_step = -1
    try:
        for epoch in range(args.epochs):
            start = resume_idx if (epoch == 0 and resume_idx > 0) else 0
            for i in range(start, len(train_examples)):
                run_step(train_examples[i])
                # 样本级进度（卡死时可定位到样本位置）
                if (i + 1) % 100 == 0:
                    logger.info(f"  [sample] {i+1}/{len(train_examples)} (global_step={global_step})")
                # 只在 global_step 变化（跨过save边界）时保存一次，避免重复保存
                if (args.save_steps > 0 and global_step > 0
                        and global_step % args.save_steps == 0 and global_step != last_saved_step):
                    save_checkpoint(output_dir, model, opt, scheduler, global_step, i + 1)
                    last_saved_step = global_step
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # 保存已有进度（可用 --resume 续训）
        if global_step > 0:
            save_checkpoint(output_dir, model, opt, scheduler, global_step, resume_idx)
        raise

    # 处理末尾不足一个accum的梯度
    if accum_count > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        opt.zero_grad()
        global_step += 1

    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed/60:.1f} min ({global_step} steps)")

    save_path = output_dir / "final_adapter"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Adapter saved to {save_path}")


if __name__ == "__main__":
    main()
