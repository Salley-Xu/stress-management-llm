"""
SFT-Stage2 训练数据池构建脚本

混合三部分（对应步骤10：多轮训练集V2 + 安全专项训练集V2 + Stage1 replay防退化）:
  1. 多轮训练集V2：DeepSeek生成（M1-M6），训练前文保持/策略调整/状态感知
  2. 安全专项训练集V2：DeepSeek生成（S1-S6），上采样×2（重点解决风险漏检）
  3. Stage1 replay：从sft_stage1_subset抽样，防止单轮能力和自然度退化

用法:
  python data_processing/build_sft2_pool.py
"""

import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_dir", default="data/processed/sft2_data", help="多轮/安全数据目录")
    parser.add_argument("--replay_file", default="data/processed/sft_stage1_subset.jsonl")
    parser.add_argument("--replay_size", type=int, default=800)
    parser.add_argument("--safety_upsample", type=int, default=2, help="安全专项上采样倍数")
    parser.add_argument("--output", default=None, help="输出文件（默认自动命名）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    multi_dir = Path(args.multi_dir)
    multi_files = sorted(multi_dir.glob("multi_turn_*.jsonl"))
    safety_files = sorted(multi_dir.glob("safety_*.jsonl"))
    if not multi_files:
        logger.error("No multi_turn_*.jsonl found. Run generate_sft2_data.py --mode multi first.")
        return
    if not safety_files:
        logger.error("No safety_*.jsonl found. Run generate_sft2_data.py --mode safety first.")
        return

    # 合并所有批次（并发生成会产出多个文件）
    multi = []
    for f in multi_files:
        multi += load_jsonl(f)
    safety = []
    for f in safety_files:
        safety += load_jsonl(f)
    logger.info(f"多轮V2: {len(multi)} 条 ({len(multi_files)} 个批次)")
    logger.info(f"安全V2: {len(safety)} 条 ({len(safety_files)} 个批次)")

    # 安全专项上采样
    safety_pool = safety * args.safety_upsample
    logger.info(f"安全专项上采样×{args.safety_upsample}: {len(safety)} → {len(safety_pool)}")

    # Stage1 replay 抽样（固定种子，可复现）
    replay_src = load_jsonl(args.replay_file)
    replay = random.sample(replay_src, min(args.replay_size, len(replay_src)))
    logger.info(f"Stage1 replay: 抽样 {len(replay)} 条 (来自 {len(replay_src)})")

    # 标记来源
    for r in multi:
        r["_meta"]["stage2_role"] = "multi_turn_v2"
    for r in safety_pool:
        r["_meta"]["stage2_role"] = "safety_v2"
    for r in replay:
        r["_meta"]["stage2_role"] = "stage1_replay"

    pool = multi + safety_pool + replay
    random.shuffle(pool)
    logger.info(f"混合池合计: {len(pool)} 条 (multi={len(multi)} + safety={len(safety_pool)} + replay={len(replay)})")

    output = args.output or f"data/processed/sft2_data/sft2_pool_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Output: {output}")


if __name__ == "__main__":
    main()
