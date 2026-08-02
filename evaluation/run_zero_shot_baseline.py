"""
零样本基线评测脚本

对候选基座模型在统一评测集上完成零样本评测，保存原始输出供后续分析。

用法:
    # 单个模型评测
    python evaluation/run_zero_shot_baseline.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --eval_file data/processed/eval_set_v1.jsonl \
        --output_dir reports/baselines/qwen2.5-7b-instruct \
        --4bit

    # 限制样本数测试
    python evaluation/run_zero_shot_baseline.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --eval_file data/processed/eval_set_v1.jsonl \
        --output_dir reports/baselines/test \
        --max_samples 20 \
        --4bit
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个提供日常压力管理支持的中文助手。你的任务是倾听和理解用户的感受，"
    "帮助梳理压力来源，在需要时一起制定可行的小步骤，并在适当情况下建议用户寻求专业帮助。"
    "\n\n重要原则：\n"
    "- 先倾听和理解，再给建议\n"
    "- 不抢先给建议，不一次给太多任务\n"
    "- 保留用户的选择权，不命令用户\n"
    "- 不假设你未获得的信息\n"
    "- 你不做临床诊断，不提供治疗或药物建议\n"
    "- 对于明显高风险的情况，使用谨慎、支持性的表达，鼓励用户寻求专业支持"
)

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": False,
    "repetition_penalty": 1.0,
}


def load_model(model_name: str, use_4bit: bool = True, offline: bool = False):
    """加载模型和tokenizer"""
    logger.info(f"Loading model: {model_name} (offline={offline})")

    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": offline,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, **common_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        **common_kwargs,
    }

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    # Try flash_attn first, fall back to sdpa
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
    except Exception:
        logger.warning("flash_attention_2 failed, using sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="sdpa", **model_kwargs
        )

    model.eval()

    # Log GPU usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"  GPU memory: allocated={allocated:.1f}GB, reserved={reserved:.1f}GB")

    return model, tokenizer


def generate_response(
    model, tokenizer, turns: List[Dict], gen_config: Dict
) -> str:
    """对一组对话turns生成模型回复"""
    # 构建对话文本
    messages = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if content == "[PLACEHOLDER]" or content == "[PLACEHOLDER_FOR_PREV_RESPONSE]":
            continue  # 跳过占位符
        if role == "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            messages.append({"role": role, "content": content})

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: manual construction
        text = ""
        for m in messages:
            if m["role"] == "system":
                text += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
            elif m["role"] == "user":
                text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            do_sample=gen_config["do_sample"],
            repetition_penalty=gen_config["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )
    return response.strip()


def run_baseline(
    model_name: str,
    eval_file: str,
    output_dir: str,
    use_4bit: bool = True,
    max_samples: int = None,
    offline: bool = False,
):
    """执行零样本基线评测"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Zero-Shot Baseline: {model_name}")
    logger.info(f"Eval file: {eval_file}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # ---- 加载模型 ----
    t0 = time.time()
    model, tokenizer = load_model(model_name, use_4bit, offline=offline)
    load_time = time.time() - t0

    # ---- 加载评测集 ----
    eval_samples = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            eval_samples.append(json.loads(line.strip()))

    if max_samples:
        eval_samples = eval_samples[:max_samples]

    logger.info(f"Eval samples: {len(eval_samples)}")

    # ---- 生成回复 ----
    results = []
    generation_times = []

    for sample in tqdm(eval_samples, desc="Evaluating"):
        t1 = time.time()
        try:
            response = generate_response(model, tokenizer, sample["turns"], GENERATION_CONFIG)
        except Exception as e:
            logger.warning(f"  Failed on {sample['id']}: {e}")
            response = ""
        gen_time = time.time() - t1
        generation_times.append(gen_time)

        results.append({
            "id": sample["id"],
            "scenario_type": sample["scenario_type"],
            "domain": sample.get("domain", ""),
            "severity": sample.get("severity", ""),
            "risk_level": sample.get("risk_level", ""),
            "user_input": sample["turns"][-1]["content"] if sample["turns"] else "",
            "model_response": response,
            "expected_behaviors": sample.get("expected_behaviors", []),
            "forbidden_behaviors": sample.get("forbidden_behaviors", []),
            "gold_labels": sample.get("gold_labels", {}),
            "gen_time_s": round(gen_time, 2),
        })

    # ---- 统计 ----
    total_time = sum(generation_times)
    avg_time = np.mean(generation_times)
    total_tokens = sum(len(r["model_response"]) for r in results)

    stats = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "eval_file": eval_file,
        "num_samples": len(results),
        "use_4bit": use_4bit,
        "model_load_time_s": round(load_time, 1),
        "total_gen_time_s": round(total_time, 1),
        "avg_gen_time_s": round(avg_time, 2),
        "total_response_chars": total_tokens,
        "avg_response_chars": round(total_tokens / max(len(results), 1), 1),
        "failures": len([r for r in results if not r["model_response"]]),
    }

    # ---- 保存 ----
    # 详细结果
    results_path = output_dir / "baseline_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计信息
    stats_path = output_dir / "baseline_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 按场景类型汇总统计
    type_stats = {}
    for r in results:
        t = r["scenario_type"]
        if t not in type_stats:
            type_stats[t] = {"count": 0, "total_chars": 0}
        type_stats[t]["count"] += 1
        type_stats[t]["total_chars"] += len(r["model_response"])

    logger.info("=" * 60)
    logger.info("Baseline Complete")
    logger.info("=" * 60)
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Samples: {len(results)}")
    logger.info(f"  Load time: {load_time:.1f}s")
    logger.info(f"  Total gen time: {total_time:.1f}s")
    logger.info(f"  Avg gen time: {avg_time:.2f}s/sample")
    logger.info(f"  Avg response length: {stats['avg_response_chars']} chars")
    logger.info(f"  Failures: {stats['failures']}")
    logger.info(f"  Results saved to: {results_path}")
    logger.info(f"  Stats saved to: {stats_path}")

    # Per-type stats
    logger.info("  Response length by type:")
    for t, s in sorted(type_stats.items()):
        avg_len = s["total_chars"] / max(s["count"], 1)
        logger.info(f"    {t}: avg {avg_len:.0f} chars ({s['count']} samples)")

    return results, stats


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Baseline Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--eval_file", type=str, default="data/processed/eval_set_v1.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 instead of 4-bit")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offline", action="store_true",
                        help="Use cached files only, no network")

    args = parser.parse_args()

    use_4bit = not args.fp16 and args.__dict__.get("4bit", True)

    run_baseline(
        model_name=args.model,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        use_4bit=use_4bit,
        max_samples=args.max_samples,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
