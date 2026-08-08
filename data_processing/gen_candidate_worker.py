"""
候选生成子进程worker：单模型独立进程，加载模型→生成→退出
（独立进程确保GPU显存完全释放，避免8GB显存下模型切换失败）

用法:
  python data_processing/gen_candidate_worker.py --model base|stage1|stage2 \
      --prompts prompts.jsonl --out cand_{model}.jsonl
"""

import json
import sys
import time
import argparse
import logging

sys.path.insert(0, "data_processing")
from build_preference_data import (
    load_model_for_gen, gen_response, STAGE1_ADAPTER, STAGE2_ADAPTER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ADAPTERS = {"base": None, "stage1": STAGE1_ADAPTER, "stage2": STAGE2_ADAPTER}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "stage1", "stage2"], required=True)
    parser.add_argument("--prompts", required=True, help="prompts jsonl")
    parser.add_argument("--out", required=True, help="输出jsonl")
    args = parser.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts, encoding="utf-8")]
    logger.info(f"[{args.model}] loading model, {len(prompts)} prompts")
    model, tokenizer = load_model_for_gen(ADAPTERS[args.model])

    t0 = time.time()
    results = []
    for i, p in enumerate(prompts):
        try:
            resp = gen_response(model, tokenizer, p["turns"])
        except Exception as e:
            logger.warning(f"  {p['prompt_id']} failed: {e}")
            resp = ""
        results.append({"prompt_id": p["prompt_id"], "response": resp})
        if (i + 1) % 50 == 0:
            logger.info(f"  [{args.model}] {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"[{args.model}] DONE, {len(results)} responses -> {args.out}")


if __name__ == "__main__":
    main()
