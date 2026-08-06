"""
专家数据标签规范化脚本

将专家数据的 Rubric 风格标签映射到项目标准标签体系：
  stress_type -> DS-* (压力领域)
  user_intent -> UG-* (用户目标)
  strategy    -> ST-* (支持策略)

用法: python data_processing/normalize_expert_labels.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/expert_data/expert_pool_v1.jsonl")
OUTPUT_PATH = Path("data/processed/expert_data/expert_pool_v1_norm.jsonl")

# stress_type -> DS-*
STRESS_TYPE_MAP = {
    "academic": "DS-LRN",
    "work": "DS-WRK",
    "career": "DS-CAR",
    "interpersonal": "DS-INT",
    "relationship": "DS-REL",
    "family": "DS-FAM",
    "financial": "DS-FIN",
    "sleep": "DS-SLP",
    "migration": "DS-MIG",
    "procrastination": "DS-PRC",
}

# user_intent -> UG-*
USER_INTENT_MAP = {
    "emotional_support": "UG-HRD",
    "problem_exploration": "UG-CLA",
    "advice_seeking": "UG-DEC",
    "action_planning": "UG-PLN",
    "high_risk": "UG-HRD",  # 高风险场景目标复杂，归为情绪支持
}

# strategy -> ST-*
STRATEGY_MAP = {
    "empathy": "ST-RFL",
    "exploration": "ST-OPN",
    "cognitive_reframe": "ST-REF",
    "action_planning": "ST-MIC",
    "risk_handling": "ST-SAF",
}


def normalize_sample(sample: dict) -> dict:
    """规范化单条样本的标签"""
    labels = sample.get("labels", {})

    # 领域映射：只保留能映射到标准DS-*标签的值
    domains = []
    for d in labels.get("domains", []):
        mapped = STRESS_TYPE_MAP.get(d)
        if mapped and mapped not in domains:
            domains.append(mapped)
    if not domains:
        domains = ["DS-INT"]
    labels["domains"] = domains

    # 用户目标映射：只保留能映射到标准UG-*标签的值
    goals = []
    for g in labels.get("user_goals", []):
        mapped = USER_INTENT_MAP.get(g)
        if mapped and mapped not in goals:
            goals.append(mapped)
    if not goals:
        goals = ["UG-CLA"]
    labels["user_goals"] = goals

    # 策略映射：只保留能映射到标准ST-*标签的值，其他丢弃
    strategies = []
    for st in labels.get("strategies", []):
        mapped = STRATEGY_MAP.get(st)
        if mapped and mapped not in strategies:
            strategies.append(mapped)
    # 如果为空，给默认策略
    if not strategies:
        strategies = ["ST-RFL", "ST-VAL"]
    labels["strategies"] = strategies

    sample["labels"] = labels
    return sample


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            normalized = normalize_sample(sample)
            f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Normalized: {count} samples -> {OUTPUT_PATH}")

    # 打印示例
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            s = json.loads(line)
            logger.info(f"  Sample: domains={s['labels']['domains']}, goals={s['labels']['user_goals']}, strategies={s['labels']['strategies']}")


if __name__ == "__main__":
    main()
