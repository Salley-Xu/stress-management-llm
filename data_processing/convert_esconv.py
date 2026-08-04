"""
ESConv / ESConv-LLM 格式转换脚本

将 ESConv 数据集转换为项目统一 SFT schema (data_schema_v1.md)

ESConv 支持的策略标签映射:
    Question -> ST-OPN (开放式澄清)
    Restatement or Paraphrasing -> ST-SUM (总结复述)
    Reflection of feelings -> ST-RFL (情绪反映)
    Self-disclosure -> (不映射，本项目不使用)
    Affirmation and Reassurance -> ST-VAL (确认接纳)
    Providing Suggestions -> ST-MIC (微行动) 或 ST-PRB (问题解决)
    Information -> ST-PRB
    Others -> (根据内容判断)

用法: python data_processing/convert_esconv.py
"""

import json
import os
from datetime import datetime
from datasets import load_dataset

# ESConv strategy -> Our strategy mapping
STRATEGY_MAP = {
    "Question": ["ST-OPN"],
    "Restatement or Paraphrasing": ["ST-SUM"],
    "Reflection of feelings": ["ST-RFL"],
    "Affirmation and Reassurance": ["ST-VAL"],
    "Providing Suggestions": ["ST-MIC"],
    "Information": ["ST-PRB"],
    "Self-disclosure": [],  # skip - we don't use this
    "Others": [],
}

# ESConv emotion -> Our domain mapping (approximate)
EMOTION_DOMAIN_MAP = {
    "anxiety": "DS-WRK",  # work/study anxiety
    "depression": "DS-SLP",  # linked to sleep/mood
    "loneliness": "DS-INT",
    "anger": "DS-INT",
    "fear": "DS-CAR",
    "sadness": "DS-REL",
    "jealousy": "DS-REL",
    "guilt": "DS-FAM",
    "disappointment": "DS-CAR",
    "disgust": "DS-INT",
}

# Emotion -> Severity mapping (rough estimate)
EMOTION_SEVERITY_MAP = {
    "anxiety": "SV-MOD",
    "depression": "SV-PER",
    "loneliness": "SV-PER",
    "sadness": "SV-MOD",
    "fear": "SV-MOD",
    "anger": "SV-MOD",
    "jealousy": "SV-MOD",
    "guilt": "SV-MOD",
    "disappointment": "SV-MLD",
    "disgust": "SV-MLD",
}


def deduce_domain(conversations: list) -> list:
    """从对话内容推断压力领域（简单关键词匹配）"""
    full_text = " ".join([c.get("content", "") for c in conversations]).lower()
    domains = []

    keywords = {
        "DS-WRK": ["work", "job", "boss", "colleague", "career", "office", "project", "deadline"],
        "DS-LRN": ["exam", "study", "school", "college", "university", "grade", "homework"],
        "DS-REL": ["boyfriend", "girlfriend", "wife", "husband", "breakup", "marriage", "date"],
        "DS-FAM": ["parent", "mother", "father", "family", "child", "mom", "dad"],
        "DS-INT": ["friend", "roommate", "lonely", "alone", "social"],
        "DS-FIN": ["money", "debt", "financial", "rent", "bills", "afford"],
        "DS-SLP": ["sleep", "insomnia", "tired", "fatigue", "exhausted"],
        "DS-CAR": ["interview", "hire", "fired", "unemployed", "resume"],
    }

    for domain, kws in keywords.items():
        if any(kw in full_text for kw in kws):
            domains.append(domain)

    if not domains:
        domains = ["DS-INT"]  # default fallback

    return domains[:2]  # max 2 domains


def convert_esconv_to_schema(sample: dict, sample_id: int) -> dict:
    """将一条 ESConv-LLM 样本转换为项目 schema"""
    conversations = sample["conversations"]
    emotion = sample.get("emotion_type", "anxiety")
    problem = sample.get("problem_type", "")

    # 构建 messages
    messages = []
    strategies_used = []

    for turn in conversations:
        role = "user" if turn["role"] == "user" else "assistant"
        content = turn["content"]
        messages.append({"role": role, "content": content})

        # 收集 assistant 使用的策略
        if role == "assistant":
            strategy = turn.get("strategy", "Others")
            mapped = STRATEGY_MAP.get(strategy, [])
            if mapped:
                strategies_used.extend(mapped)

    # 如果对话最后不是 assistant，截断
    if messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]

    if len(messages) < 2:
        return None

    # 构建 schema
    result = {
        "_meta": {
            "id": f"esconv_{sample_id:05d}",
            "source": "public",
            "source_name": "ESConv-LLM (thu-coai/esconv via Estwld/esconv_llm)",
            "license": "Apache-2.0",
            "generation_method": "human_written",
            "annotator": "auto",
            "review_status": "pending",
            "version": "1.0",
            "created_date": "2026-08-04",
            "usage": "train",
            "language": "en",
        },
        "type": "sft_single" if len(messages) <= 6 else "sft_multiturn",
        "labels": {
            "domains": deduce_domain(conversations),
            "severity": EMOTION_SEVERITY_MAP.get(emotion, "SV-MOD"),
            "user_goals": ["UG-CLA"],
            "strategies": strategies_used[:6] if strategies_used else ["ST-RFL", "ST-VAL"],
        },
        "messages": messages,
        "quality_labels": {
            "errors": [],
            "notes": f"From ESConv: emotion={emotion}, problem={problem}. English, needs translation."
        }
    }
    return result


def main():
    print("Loading ESConv-LLM dataset...")
    ds = load_dataset("Estwld/esconv_llm", split="train")

    output_dir = "data/processed/public_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "esconv_sft_v1.jsonl")

    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(ds):
            converted = convert_esconv_to_schema(sample, i)
            if converted:
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    print(f"Converted: {count} samples -> {output_path}")
    print(f"Skipped: {skipped}")
    print(f"\nSample output (first 2):")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            sample = json.loads(line)
            print(f"\n--- Sample {sample['_meta']['id']} ---")
            print(f"  Domains: {sample['labels']['domains']}")
            print(f"  Severity: {sample['labels']['severity']}")
            print(f"  Strategies: {sample['labels']['strategies']}")
            print(f"  Turns: {len(sample['messages'])}")
            print(f"  Language: {sample['_meta']['language']}")


if __name__ == "__main__":
    main()
