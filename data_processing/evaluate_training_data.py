"""
训练数据质量评估脚本

按来源分层抽样，用 DeepSeek 按 100 分制评分标准评估训练数据质量。

100分制7维度:
  输入质量(15) + 场景真实性(15) + 回复质量(20) + 个性化(15)
  + 行动价值(15) + 安全性(15) + 风格一致性(5) = 100

数据筛选:
  90-100 核心SFT / 80-90 正常SFT / 70-80 低权重
  60-70 DPO rejected / <60 删除

用法:
  # 每来源抽100条评估
  python data_processing/evaluate_training_data.py --per_source 100
"""

import json
import os
import sys
import time
import logging
import argparse
import requests
from datetime import datetime
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

SCORING_PROMPT = """你是一位训练数据质量评审专家。请根据以下100分制评分标准，对一段压力管理对话数据评分。

【用户输入】
{user_msg}

【模型回复】
{assistant_msg}

【评分维度】（满分100）
1. 输入质量(15分)：用户输入真实性、完整性、训练价值
2. 场景真实性(15分)：是否符合真实压力场景
3. 回复质量(20分)：理解程度、回应质量、支持合理性
4. 个性化程度(15分)：是否针对用户具体情况回应
5. 行动价值(15分)：建议是否具体可执行
6. 安全性(15分)：是否符合安全规范
7. 风格一致性(5分)：是否温和、专业、非说教

输出JSON格式：
{{
  "input_quality": 分数,
  "scene_realism": 分数,
  "response_quality": 分数,
  "personalization": 分数,
  "action_value": 分数,
  "safety": 分数,
  "style_consistency": 分数,
  "total": 总分(0-100),
  "category": "core|normal|low_weight|dpo_rejected|delete"
}}"""


def call_deepseek(prompt: str, api_key: str = None, max_tokens: int = 1024) -> str:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # 评分用低温度，更稳定
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(4):  # 网络重试
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=90)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(2 * (attempt + 1))  # 递增退避
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


def classify(total: int, safety: int) -> str:
    """按总分+安全性硬规则分类"""
    if safety <= 7:
        return "delete"
    if total >= 90:
        return "core"
    if total >= 80:
        return "normal"
    if total >= 70:
        return "low_weight"
    if total >= 60:
        return "dpo_rejected"
    return "delete"


def evaluate_sample(sample: dict, api_key: str) -> dict:
    """评估单条样本"""
    messages = sample.get("messages", [])
    if len(messages) < 2:
        return {"status": "error", "error": "invalid messages"}

    # 取用户输入（第一条user）和模型回复（最后一条assistant）
    user_msg = ""
    assistant_msg = ""
    for m in messages:
        if m["role"] == "user" and not user_msg:
            user_msg = m["content"]
        if m["role"] == "assistant":
            assistant_msg = m["content"]

    if not user_msg or not assistant_msg:
        return {"status": "error", "error": "missing content"}

    prompt = SCORING_PROMPT.format(user_msg=user_msg[:500], assistant_msg=assistant_msg[:500])

    score_data = {}
    for attempt in range(3):
        try:
            text = call_deepseek(prompt, api_key=api_key)
            score_data = parse_json(text)
            if score_data and "total" in score_data:
                break
        except Exception:
            pass

    if not score_data or "total" not in score_data:
        return {"status": "error", "error": "scoring failed"}

    return {
        "status": "ok",
        "source": sample.get("_meta", {}).get("source", "unknown"),
        "source_name": sample.get("_meta", {}).get("source_name", ""),
        "domain": sample.get("labels", {}).get("domains", [""])[0],
        "difficulty_type": sample.get("labels", {}).get("difficulty_type", ""),
        "rubric_category": sample.get("_meta", {}).get("rubric_category", ""),
        "score": score_data,
        "category": classify(score_data.get("total", 0), score_data.get("safety", 15)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_source", type=int, default=50, help="每来源抽样数")
    parser.add_argument("--input", type=str, default="data/processed/cleaned_data_v1.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/processed/eval_reports")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("No DEEPSEEK_API_KEY")
        sys.exit(1)

    # 加载数据
    with open(args.input, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Loaded {len(samples)} samples from {args.input}")

    # 按来源分层
    sources = {}
    for s in samples:
        sn = s.get("_meta", {}).get("source_name", "unknown")
        # 归类: SmileChat / 合成 / 专家
        if "SmileChat" in sn:
            key = "smilechat"
        elif "expert" in sn:
            key = "expert"
        else:
            key = "synthetic"
        sources.setdefault(key, []).append(s)

    for k, v in sources.items():
        logger.info(f"  {k}: {len(v)} samples")

    # 抽样
    import random
    random.seed(42)
    eval_samples = []
    for k, v in sources.items():
        sample_count = min(args.per_source, len(v))
        eval_samples.extend(random.sample(v, sample_count))
    logger.info(f"Eval samples: {len(eval_samples)}")

    # 评估
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"quality_report_{ts}.jsonl"

    results = []
    errors = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(eval_samples):
            result = evaluate_sample(sample, api_key)
            if result["status"] == "ok":
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                errors += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  Progress: {i+1}/{len(eval_samples)} ({rate:.1f}/s)")

    logger.info(f"Evaluated: {len(results)} OK, {errors} errors")
    logger.info(f"Results saved to {out_path}")

    # 汇总统计
    if results:
        from collections import Counter
        cats = Counter(r["category"] for r in results)
        by_source = Counter((r["source_name"], r["category"]) for r in results)
        avg_total = sum(r["score"].get("total", 0) for r in results) / len(results)
        avg_safety = sum(r["score"].get("safety", 0) for r in results) / len(results)

        logger.info("=" * 50)
        logger.info(f"Average total score: {avg_total:.1f}/100")
        logger.info(f"Average safety score: {avg_safety:.1f}/15")
        logger.info(f"Category distribution: {dict(cats)}")
        logger.info("By source:")
        for (src, cat), cnt in sorted(by_source.items()):
            logger.info(f"  {src}: {cat}={cnt}")


if __name__ == "__main__":
    main()
