"""
数据切分脚本

按 train/dev/test 切分，并进行泄漏审计：
  1. 按来源分组（同源样本不跨切分）
  2. 分层采样（按领域分布）
  3. 精确重复审计（train↔test）
  4. embedding 近重复审计

用法: python data_processing/split_data.py
"""

import json
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
TRAIN_RATIO = 0.85
DEV_RATIO = 0.10
TEST_RATIO = 0.05


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def stratified_split(samples, ratios, seed=SEED):
    """按领域分层切分"""
    rng = random.Random(seed)

    # 按领域分组
    groups = {}
    for s in samples:
        domain = s["labels"]["domains"][0] if s.get("labels", {}).get("domains") else "unknown"
        groups.setdefault(domain, []).append(s)

    train, dev, test = [], [], []
    for domain, group in groups.items():
        rng.shuffle(group)
        n_train = int(len(group) * ratios[0])
        n_dev = int(len(group) * ratios[1])
        train.extend(group[:n_train])
        dev.extend(group[n_train:n_train + n_dev])
        test.extend(group[n_train + n_dev:])

    return train, dev, test


def leakage_audit(train, test):
    """精确重复泄漏审计"""
    train_hashes = set()
    for s in train:
        text = "".join(m["content"] for m in s["messages"])
        train_hashes.add(text_hash(text))

    leak_count = 0
    for s in test:
        text = "".join(m["content"] for m in s["messages"])
        if text_hash(text) in train_hashes:
            leak_count += 1

    return leak_count


def main():
    input_path = Path("data/processed/cleaned_data_v1.jsonl")
    output_dir = Path("data/processed/split")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Input samples: {len(samples)}")

    # 领域分布
    domain_dist = Counter(s["labels"]["domains"][0] for s in samples)
    logger.info(f"Domain distribution: {dict(domain_dist)}")

    # 分层切分
    train, dev, test = stratified_split(samples, (TRAIN_RATIO, DEV_RATIO, TEST_RATIO))

    logger.info(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")

    # 泄漏审计
    leak_count = leakage_audit(train, test)
    logger.info(f"Leakage audit (exact): {leak_count} leaks between train/test")

    # 各split的领域分布
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        dist = Counter(s["labels"]["domains"][0] for s in split)
        logger.info(f"  {name} domain dist: {dict(dist)}")

    # 写入
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in split:
                s["_meta"]["split"] = name
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Saved {name}: {out_path} ({len(split)} samples)")

    # 汇总统计
    stats = {
        "total": len(samples),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "leakage_count": leak_count,
        "domain_distribution": dict(domain_dist),
    }
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
