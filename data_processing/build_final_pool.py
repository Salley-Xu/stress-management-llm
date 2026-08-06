"""
重建最终训练数据池（混合策略）

质量评估结论（300条抽样）:
  - 专家数据(491): 合格率99% → 全部保留
  - DeepSeek合成(5.4K): 合格率98% → 全部保留
  - SmileChat(37K): 合格率仅8.5% → 随机抽样3K作为补充

用法: python data_processing/build_final_pool.py
"""

import json
import random
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/cleaned_data_v1.jsonl")
OUTPUT_PATH = Path("data/processed/final_pool_v1.jsonl")

SMILE_SAMPLE = 3000
SEED = 42


def classify_source(sample: dict) -> str:
    """按来源分类"""
    sn = sample.get("_meta", {}).get("source_name", "")
    if "expert" in sn:
        return "expert"
    if "SmileChat" in sn:
        return "smilechat"
    return "synthetic"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Input: {len(samples)} samples")

    # 分类
    by_source = {"expert": [], "synthetic": [], "smilechat": []}
    for s in samples:
        by_source[classify_source(s)].append(s)

    for k, v in by_source.items():
        logger.info(f"  {k}: {len(v)}")

    # 混合策略
    rng = random.Random(SEED)
    final = []

    # 专家+合成全保留
    final.extend(by_source["expert"])
    final.extend(by_source["synthetic"])

    # SmileChat随机抽样
    smile = by_source["smilechat"]
    smile_sample = rng.sample(smile, min(SMILE_SAMPLE, len(smile)))
    final.extend(smile_sample)

    logger.info(f"Final pool: {len(final)} samples")
    logger.info(f"  expert={len(by_source['expert'])}, synthetic={len(by_source['synthetic'])}, smilechat_sample={len(smile_sample)}")

    # 打乱
    rng.shuffle(final)

    # 写入
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {OUTPUT_PATH}")

    # 领域分布
    domain_dist = Counter(s["labels"]["domains"][0] for s in final)
    logger.info(f"Domain distribution: {dict(domain_dist)}")


if __name__ == "__main__":
    main()
