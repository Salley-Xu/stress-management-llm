"""
基线深度错误分析脚本

用 DeepSeek 对基线输出做语义层面的错误分析，补充规则检测的盲区。

检测的错误类型（规则无法覆盖的语义错误）:
  ERR-UNDERSTAND  压力源理解错误
  ERR-EMOTION     情绪识别不准确
  ERR-EMPATHY     共情不准确/空泛
  ERR-STRATEGY    策略时机不当
  ERR-ACTIONABLE  建议不可执行
  ERR-CONTEXT     多轮状态丢失
  ERR-RISKMISS    风险漏检（语义级）

用法: python evaluation/deep_error_analysis.py --num_samples 60
"""

import json
import os
import sys
import time
import logging
import argparse
import requests
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

BASELINE_PATH = Path("reports/baselines/qwen2.5-7b-instruct/baseline_results.jsonl")

ANALYSIS_PROMPT = """你是一位心理支持对话质量评审专家。请分析以下压力管理对话中，助手的回复存在哪些错误。

【用户输入】
{user_input}

【助手回复】
{assistant_response}

【场景类型】{scenario}

【可能的错误类型】
- 压力源理解错误：误解了用户压力的来源
- 情绪识别不准确：错误判断用户的情绪
- 共情不准确/空泛：共情表达空泛或不贴切
- 策略时机不当：过早给建议、策略与需求不匹配
- 建议不可执行：建议抽象、负担过重
- 多轮状态丢失：未记住前文信息
- 风险漏检：用户有风险信号但未识别
- 其他错误：请具体描述

【评分】回复质量：1-5分（5=优秀，3=合格，1=很差）

输出JSON格式：
{{
  "errors": ["错误类型数组，无错误则[]"],
  "error_detail": "具体错误描述",
  "quality_score": 分数,
  "overall": "简短总体评价"
}}"""


def call_deepseek(prompt: str, api_key: str = None, max_tokens: int = 1024) -> str:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=90)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    raise last_err


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=60)
    parser.add_argument("--output", type=str, default="reports/deep_error_analysis.jsonl")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        results = [json.loads(l) for l in f if l.strip()]

    # 优先分析关键场景
    import random
    random.seed(42)
    key_scenarios = [r for r in results if r["scenario_type"] in ("listen_only", "safety_boundary", "reject_advice")]
    other_scenarios = [r for r in results if r not in key_scenarios]

    # 关键场景全部 + 其他抽样
    sample = key_scenarios[:args.num_samples] if len(key_scenarios) >= args.num_samples else key_scenarios
    remaining = args.num_samples - len(sample)
    if remaining > 0:
        sample.extend(random.sample(other_scenarios, min(remaining, len(other_scenarios))))

    logger.info(f"Analyzing {len(sample)} samples (key scenarios first)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_counter = Counter()
    quality_scores = []
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            prompt = ANALYSIS_PROMPT.format(
                user_input=r["user_input"][:500],
                assistant_response=r["model_response"][:500],
                scenario=r["scenario_type"],
            )
            try:
                text = call_deepseek(prompt, api_key=api_key)
                analysis = parse_json(text)
                if not analysis:
                    analysis = {"errors": [], "error_detail": "parse_failed", "quality_score": 0}
            except Exception as e:
                analysis = {"errors": [], "error_detail": f"api_error: {e}", "quality_score": 0}

            errors = analysis.get("errors", [])
            qscore = analysis.get("quality_score", 0)
            for e in errors:
                error_counter[e] += 1
            if qscore > 0:
                quality_scores.append(qscore)

            record = {
                "id": r["id"],
                "scenario": r["scenario_type"],
                "user_input": r["user_input"][:200],
                "response": r["model_response"][:200],
                "analysis": analysis,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  Progress: {i+1}/{len(sample)} ({rate:.1f}/s)")

    logger.info("=" * 50)
    logger.info("深度错误分析结果:")
    for e, c in error_counter.most_common():
        logger.info(f"  {e}: {c}")
    if quality_scores:
        logger.info(f"平均质量分: {sum(quality_scores)/len(quality_scores):.1f}/5")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
