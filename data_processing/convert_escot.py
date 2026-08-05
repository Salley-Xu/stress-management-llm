"""
ESCoT 格式转换脚本

将 ESCoT 数据集（GitHub: TeigenZhang/ESCoT）转换为统一 SFT schema。

ESCoT 结构:
  data/{train,val,test}.json
  [{"id", "original_data": {"dialog": [{"speaker": "seeker|supporter", "content"}],
                             "strategy": [...], "response": ...},
    "cot_data": {...}}]

角色映射:
  seeker -> user
  supporter -> assistant

策略映射（ESConv风格）:
  Question -> ST-OPN
  Restatement or Paraphrasing -> ST-SUM
  Reflection of feelings -> ST-RFL
  Affirmation and Reassurance -> ST-VAL
  Providing Suggestions -> ST-MIC
  Information -> ST-PRB

注意: ESCoT 为英文数据，转换后需翻译。本脚本先保留英文，清洗时会被语言过滤，
需后续翻译后重新入库。

用法: python data_processing/convert_escot.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/ESCoT/data")
OUTPUT_PATH = Path("data/processed/public_data/escot_sft_v1.jsonl")

STRATEGY_MAP = {
    "Question": ["ST-OPN"],
    "Restatement or Paraphrasing": ["ST-SUM"],
    "Reflection of feelings": ["ST-RFL"],
    "Affirmation and Reassurance": ["ST-VAL"],
    "Providing Suggestions": ["ST-MIC"],
    "Information": ["ST-PRB"],
    "Self-disclosure": [],
    "Others": [],
}


def convert_sample(sample: dict, source: str) -> dict:
    """转换单条ESCoT样本"""
    orig = sample["original_data"]
    dialog = orig.get("dialog", [])

    messages = []
    strategies = []
    user_text = ""
    for turn in dialog:
        role = "user" if turn.get("speaker") == "seeker" else "assistant"
        content = turn.get("content", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})
        if role == "user":
            user_text += content

    # 合并最后一条 assistant 回复（strategy字段对应）
    if orig.get("strategy"):
        for s in orig["strategy"]:
            mapped = STRATEGY_MAP.get(s, [])
            if mapped:
                strategies.extend(mapped)

    if len(messages) < 2:
        return None
    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if len(messages) < 2:
        return None

    return {
        "_meta": {
            "id": f"escot_{sample['id']}",
            "source": "public",
            "source_name": f"ESCoT ({source})",
            "license": "MIT",
            "generation_method": "llm_synthetic",
            "review_status": "pending",
            "version": "1.0",
            "created_date": "2026-08-05",
            "usage": "train",
            "language": "en",
        },
        "type": "sft_single" if len(messages) <= 6 else "sft_multiturn",
        "labels": {
            "domains": ["DS-INT"],
            "severity": "SV-MOD",
            "user_goals": ["UG-CLA"],
            "strategies": strategies[:8] if strategies else ["ST-RFL", "ST-VAL"],
        },
        "messages": messages,
        "quality_labels": {
            "errors": [],
            "notes": f"ESCoT英文，需翻译。含cot_data在原始文件。strategy={orig.get('strategy')}"
        }
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for split in ["train", "val", "test"]:
            filepath = RAW_DIR / f"{split}.json"
            if not filepath.exists():
                logger.warning(f"Missing: {filepath}")
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            count = 0
            for sample in data:
                converted = convert_sample(sample, split)
                if converted:
                    out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
            logger.info(f"{split}: {count} samples")
            all_count += count

    logger.info(f"Total ESCoT converted: {all_count} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
