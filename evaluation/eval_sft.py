"""
SFT后模型评测脚本

加载 base 模型 + SFT adapter，在评测集上生成回复，保存结果供LLM错误分析。

用法:
  python evaluation/eval_sft.py --adapter checkpoints/sft_stage1/final_adapter \
      --output_dir reports/baselines/sft_stage1
"""

import json
import time
import logging
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "D:/anaconda3/envs/stress-mgmt/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"

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


def load_model_with_adapter(adapter_path: str):
    """加载base模型 + SFT adapter"""
    logger.info(f"Loading base model: {MODEL_PATH}")
    logger.info(f"Loading adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU memory: {allocated:.1f}GB")
    return model, tokenizer


def generate_response(model, tokenizer, turns):
    """生成回复（与基线脚本一致）"""
    messages = []
    for turn in turns:
        role = turn["role"]
        content = turn["content"]
        if content.startswith("[PLACEHOLDER"):
            continue
        if role == "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        else:
            messages.append({"role": role, "content": content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            do_sample=GENERATION_CONFIG["do_sample"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval_file", default="data/processed/eval_set_v1.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_with_adapter(args.adapter)

    with open(args.eval_file, "r", encoding="utf-8") as f:
        eval_samples = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        eval_samples = eval_samples[:args.max_samples]
    logger.info(f"Eval samples: {len(eval_samples)}")

    results = []
    gen_times = []
    t0 = time.time()

    for i, sample in enumerate(eval_samples):
        t1 = time.time()
        try:
            response = generate_response(model, tokenizer, sample["turns"])
        except Exception as e:
            logger.warning(f"  Failed on {sample['id']}: {e}")
            response = ""
        gen_times.append(time.time() - t1)

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
            "gen_time_s": round(time.time() - t1, 2),
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Progress: {i+1}/{len(eval_samples)} ({elapsed:.0f}s)")

    # 保存
    results_path = output_dir / "baseline_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "model": "sft_stage1",
        "adapter": args.adapter,
        "num_samples": len(results),
        "total_time_s": round(time.time() - t0, 1),
        "avg_gen_time_s": round(sum(gen_times) / len(gen_times), 2),
        "failures": len([r for r in results if not r["model_response"]]),
        "avg_response_chars": round(sum(len(r["model_response"]) for r in results) / len(results), 1),
    }
    with open(output_dir / "baseline_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
