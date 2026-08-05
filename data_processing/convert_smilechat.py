"""
SmileChat 格式转换脚本

将 SmileChat 数据集（GitHub: qiuhuachuan/smile）转换为统一 SFT schema。

SmileChat 结构:
  data/*.json 每个文件是一个对话
  [{"role": "client", "content": "...", "annotation": [...]},
   {"role": "counselor", "content": "..."}, ...]

角色映射:
  client -> user
  counselor -> assistant

用法: python data_processing/convert_smilechat.py
"""

import json
import re
import glob
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/smile/data")
OUTPUT_PATH = Path("data/processed/public_data/smilechat_sft_v1.jsonl")


def deduce_domain(text: str) -> list:
    """从对话内容推断压力领域"""
    domains = []
    keywords = {
        "DS-LRN": ["考试", "复习", "学习", "论文", "考研", "成绩", "学业", "读书", "上课", "作业", "大学", "老师", "导师"],
        "DS-WRK": ["工作", "加班", "老板", "同事", "上班", "项目", "领导", "职场", "职业", "升职", "辞职", "跳槽"],
        "DS-CAR": ["面试", "求职", "找工作", "简历", "失业", "转行", "offer", "招聘"],
        "DS-REL": ["男朋友", "女朋友", "对象", "分手", "恋爱", "结婚", "老公", "老婆", "感情", "伴侣", "婚姻"],
        "DS-FAM": ["父母", "妈妈", "爸爸", "母亲", "父亲", "家里", "家人", "家庭", "孩子", "儿子", "女儿", "催婚", "亲戚"],
        "DS-INT": ["朋友", "同学", "室友", "同事", "社交", "孤独", "不合群", "人际"],
        "DS-FIN": ["钱", "工资", "房贷", "房租", "债务", "经济", "收入", "存款", "还款"],
        "DS-SLP": ["失眠", "睡觉", "睡眠", "熬夜", "疲劳", "累", "没精神"],
        "DS-MIG": ["异乡", "外地", "留学生", "新城市", "搬家", "不适应"],
        "DS-PRC": ["拖延", "拖延症", "拖延症", "做不完", "没动力"],
    }
    for domain, kws in keywords.items():
        if any(kw in text for kw in kws):
            domains.append(domain)
    if not domains:
        domains = ["DS-INT"]
    return domains[:2]


def deduce_severity(text: str) -> str:
    """从对话内容推断严重度"""
    risk_words = ["自杀", "不想活", "活着没意思", "结束生命", "伤害自己", "割腕", "轻生"]
    severe_words = ["崩溃", "撑不住", "活不下去", "承受不了", "受不了了"]
    persistent_words = ["一直", "持续", "很久", "几周", "几个月", "反复", "总是"]

    if any(w in text for w in risk_words):
        return "SV-RSK"
    if any(w in text for w in severe_words):
        return "SV-IMP"
    if any(w in text for w in persistent_words):
        return "SV-PER"
    return "SV-MOD"


def deduce_goal(text: str) -> str:
    """推断用户目标"""
    if any(w in text for w in ["怎么办", "怎么做", "帮帮我", "有什么方法", "给个建议", "怎么解决"]):
        return "UG-PLN"
    if any(w in text for w in ["不知道", "想不明白", "纠结", "选择", "要不要", "该不该", "犹豫"]):
        return "UG-DEC"
    if any(w in text for w in ["难过", "难受", "烦", "焦虑", "崩溃", "委屈"]):
        return "UG-HRD"
    return "UG-CLA"


def convert_dialogue(dialogue: list, file_id: str) -> dict:
    """转换单个对话为schema"""
    messages = []
    user_text = ""
    for turn in dialogue:
        role = "user" if turn["role"] == "client" else "assistant"
        content = turn.get("content", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})
        if role == "user":
            user_text += content

    if len(messages) < 2:
        return None
    if messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if len(messages) < 2:
        return None

    return {
        "_meta": {
            "id": f"smile_{file_id}",
            "source": "public",
            "source_name": "SmileChat (qiuhuachuan/smile)",
            "license": "Apache-2.0",
            "generation_method": "llm_synthetic",  # 基于真实PsyQA扩展
            "review_status": "pending",
            "version": "1.0",
            "created_date": "2026-08-05",
            "usage": "train",
            "language": "zh-CN",
        },
        "type": "sft_single" if len(messages) <= 6 else "sft_multiturn",
        "labels": {
            "domains": deduce_domain(user_text),
            "severity": deduce_severity(user_text),
            "user_goals": [deduce_goal(user_text)],
            "strategies": ["ST-RFL", "ST-VAL"],  # 默认标签，后续精标注
        },
        "messages": messages,
        "quality_labels": {
            "errors": [],
            "notes": "SmileChat自动转换，策略标签为默认值需精标注"
        }
    }


def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.json")))
    logger.info(f"SmileChat files: {len(files)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    domain_dist = Counter()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, filepath in enumerate(files):
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    dialogue = json.load(fh)
                if not isinstance(dialogue, list):
                    skipped += 1
                    continue

                file_id = Path(filepath).stem
                converted = convert_dialogue(dialogue, file_id)
                if converted:
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
                    for d in converted["labels"]["domains"]:
                        domain_dist[d] += 1
                else:
                    skipped += 1

                if (i + 1) % 10000 == 0:
                    logger.info(f"  Progress: {i+1}/{len(files)}, converted={count}")
            except Exception as e:
                skipped += 1

    logger.info(f"Converted: {count}, skipped: {skipped}")
    logger.info(f"Domain distribution: {dict(domain_dist)}")
    logger.info(f"Saved to {OUTPUT_PATH}")

    # Print samples
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    logger.info(f"Sample (first 2):")
    for s in lines[:2]:
        logger.info(f"  {s['_meta']['id']}: domains={s['labels']['domains']}, sev={s['labels']['severity']}, turns={len(s['messages'])}")


if __name__ == "__main__":
    main()
