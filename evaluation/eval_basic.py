"""
基础评测脚本 - 对模型进行自动评测

用法:
    python evaluation/eval_basic.py \
        --model_path checkpoints/sm-sft-qwen-xxx/final_adapter \
        --eval_file data/processed/sm-eval-3k-20240801-v1/test.jsonl \
        --output_dir reports/v1-baseline \
        --metrics accuracy f1 rouge-l bert-score
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
import evaluate  # huggingface evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 加载 HF evaluate 的 BERTScore
bertscore_metric = evaluate.load("bertscore")


def load_model(model_path: str, base_model_name: str = None):
    """加载模型（支持 PEFT adapter）"""
    logger.info(f"加载模型: {model_path}")

    # 尝试作为 PEFT adapter 加载
    if base_model_name and os.path.isdir(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
            logger.info("  PEFT adapter loaded and merged.")
            return model, tokenizer
        except Exception:
            logger.warning("  PEFT 加载失败，尝试直接加载完整模型。")

    # 直接加载完整模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_response(
    model, tokenizer, messages: List[Dict], generation_config: Dict
) -> str:
    """生成模型回复"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            temperature=generation_config.get("temperature", 0.1),
            top_p=generation_config.get("top_p", 0.9),
            do_sample=generation_config.get("do_sample", False),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    )
    return response.strip()


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metric_names: List[str],
) -> Dict[str, float]:
    """计算自动评测指标"""
    results = {}

    # ROUGE-L
    if "rouge-l" in metric_names:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]
        results["rouge-l"] = round(np.mean(rouge_l_scores), 4)

    # BERTScore
    if "bert-score" in metric_names:
        bert_scores = bertscore_metric.compute(
            predictions=predictions,
            references=references,
            lang="zh",
            model_type="bert-base-chinese",
        )
        results["bert-score-f1"] = round(np.mean(bert_scores["f1"]), 4)

    # 长度一致性
    pred_lens = [len(p) for p in predictions]
    ref_lens = [len(r) for r in references]
    results["avg_pred_length"] = round(np.mean(pred_lens), 1)
    results["avg_ref_length"] = round(np.mean(ref_lens), 1)
    results["length_ratio"] = round(np.mean(pred_lens) / max(np.mean(ref_lens), 1), 2)

    return results


def run_evaluation(
    model_path: str,
    eval_file: str,
    output_dir: str,
    metric_names: List[str],
    base_model_name: Optional[str] = None,
    batch_size: int = 4,
    max_samples: int = None,
):
    """执行完整评测流程"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("基础评测")
    logger.info(f"  模型: {model_path}")
    logger.info(f"  数据: {eval_file}")
    logger.info(f"  指标: {metric_names}")
    logger.info("=" * 60)

    # ---- 加载模型 ----
    model, tokenizer = load_model(model_path, base_model_name)
    model.eval()

    # ---- 加载评测数据 ----
    dataset = load_dataset("json", data_files=eval_file, split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"  评测样本数: {len(dataset)}")

    # ---- 生成预测 ----
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": False,
    }

    predictions = []
    references = []
    results_list = []  # 详细结果

    for i, example in enumerate(tqdm(dataset, desc="Generating")):
        # 取对话历史（不含最后一轮 assistant）
        if "messages" in example:
            # 使用前面所有轮次作为历史
            context = example["messages"][:-1]  # 不含最后一轮
            reference = example["messages"][-1]["content"]
        elif "conversations" in example:
            context = example["conversations"][:-1]
            reference = example["conversations"][-1]["value"]
        else:
            context = [{"role": "user", "content": example.get("input", "")}]
            reference = example.get("output", "")

        try:
            prediction = generate_response(model, tokenizer, context, generation_config)
        except Exception as e:
            logger.warning(f"  Sample {i} generation failed: {e}")
            prediction = ""

        predictions.append(prediction)
        references.append(reference)
        results_list.append({
            "index": i,
            "prediction": prediction,
            "reference": reference,
        })

    # ---- 计算指标 ----
    metrics = compute_metrics(predictions, references, metric_names)
    logger.info("=" * 60)
    logger.info("评测结果")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    # ---- 保存 ----
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "eval_file": eval_file,
        "num_samples": len(dataset),
        "metrics": metrics,
        "generation_config": generation_config,
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"  Report saved to {report_path}")

    # 详细结果
    details_path = output_dir / "eval_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for item in results_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  Detailed results saved to {details_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Basic Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（adapter 或完整模型）")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="基座模型名称（使用 PEFT adapter 时需要）")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="评测数据 JSONL 文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--metrics", nargs="+", default=["rouge-l", "bert-score"],
                        help="评测指标列表")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制评测样本数")

    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        metric_names=args.metrics,
        base_model_name=args.base_model_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
