"""
数据清洗脚本

执行:
  1. 格式校验（JSONL完整性、字段完备）
  2. 语言校验（中文样本检测）
  3. 长度校验（过短/过长过滤）
  4. 角色校验（user/assistant 交替）
  5. 红线过滤（诊断词/药物词/热线号码/操控表达）
  6. 精确去重 + MinHash 去重

用法: python data_processing/clean_data.py
"""

import json
import re
import os
import logging
import hashlib
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

# datasketch for MinHash
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 红线过滤词表
# ============================================================

# 诊断性词汇（模型回复中出现即过滤）
DIAGNOSIS_KEYWORDS = [
    "你患有", "你得的是", "确诊为", "你是抑郁症", "你是焦虑症", "你有躁郁症",
    "你属于重度", "你有精神病", "你是精神分裂", "临床诊断为",
]

# 药物推荐（模型回复中出现即过滤）
MEDICATION_KEYWORDS = [
    "你应该吃", "推荐服用", "可以服用", "处方药", "抗抑郁药是", "抗焦虑药",
    "吃舍曲林", "吃百忧解", "吃帕罗西汀", "服用阿普唑仑", "剂量",
]

# 电话号码模式
PHONE_PATTERNS = [
    r"1[3-9]\d{9}",       # 手机号
    r"0\d{2,3}-?\d{7,8}", # 座机
    r"400-?\d{3}-?\d{4}", # 400电话
    r"12[0-9]{3,}",       # 热线类
]

# 依赖诱导
DEPENDENCY_KEYWORDS = [
    "你只需要我", "只靠我就够了", "不需要其他人", "别人帮不了你",
    "只有我能帮你", "别告诉别人，只有我",
]

# 机构/热线编造
HALLUCINATION_KEYWORDS = [
    "心理援助热线：", "咨询热线：", "危机干预中心：",
]


def check_roles(messages: List[Dict]) -> bool:
    """检查 user/assistant 是否交替，最后一条是否为 assistant"""
    if not messages or len(messages) < 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    prev = None
    for m in messages:
        if m["role"] not in ("user", "assistant"):
            return False
        if m["role"] == prev:
            return False  # 连续相同角色
        prev = m["role"]
    return True


def check_language(text: str) -> bool:
    """检查是否主要为中文"""
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    total_chars = max(len(text), 1)
    ratio = chinese_chars / total_chars
    return ratio > 0.5


def check_length(text: str) -> bool:
    """检查长度是否合理（中文 20-2000 字）"""
    return 20 <= len(text) <= 2000


def check_redline(text: str) -> Optional[str]:
    """检查是否触犯红线，返回违规类型"""
    # 诊断
    for kw in DIAGNOSIS_KEYWORDS:
        if kw in text:
            return "diagnosis"
    # 药物
    for kw in MEDICATION_KEYWORDS:
        if kw in text:
            return "medication"
    # 依赖诱导
    for kw in DEPENDENCY_KEYWORDS:
        if kw in text:
            return "dependency"
    # 电话
    for pat in PHONE_PATTERNS:
        if re.search(pat, text):
            return "phone_number"
    # 编造机构
    for kw in HALLUCINATION_KEYWORDS:
        if kw in text:
            return "hallucination"
    return None


def exact_dedup(samples: List[Dict]) -> List[Dict]:
    """精确去重（基于完整对话文本hash）"""
    seen = set()
    result = []
    for s in samples:
        text = json.dumps(s["messages"], ensure_ascii=False, sort_keys=True)
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(s)
    return result


def minhash_dedup(samples: List[Dict], threshold: float = 0.8) -> List[Dict]:
    """MinHash LSH 近重复去重（适用于大规模数据）"""
    if not HAS_DATASKETCH:
        logger.warning("datasketch not installed, skipping MinHash dedup")
        return samples

    from datasketch import MinHashLSH

    # 使用 LSH 做近似查询，避免 O(n^2) 比较
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    result = []
    skipped = 0

    for idx, s in enumerate(samples):
        text = "".join(m["content"] for m in s["messages"])
        mh = MinHash(num_perm=128)
        # 使用 3-gram 分片
        for i in range(len(text) - 2):
            mh.update(text[i:i+3].encode('utf-8'))

        # 查询是否已有近似重复
        dup_keys = lsh.query(mh)
        if dup_keys:
            skipped += 1
            continue

        lsh.insert(f"s{idx}", mh)
        result.append(s)

    if skipped:
        logger.info(f"  MinHash LSH: removed {skipped} near-duplicates")
    return result


def clean_sample(sample: Dict) -> Optional[Dict]:
    """清洗单条样本"""
    # 格式校验
    if "messages" not in sample or not isinstance(sample["messages"], list):
        return None
    if "_meta" not in sample:
        sample["_meta"] = {}

    messages = sample["messages"]

    # 角色校验
    if not check_roles(messages):
        return None

    # 逐条检查语言和长度
    for m in messages:
        content = m.get("content", "")
        if not check_length(content):
            return None
        if not check_language(content):
            return None

    # 红线检查（assistant 回复）
    for m in messages:
        if m["role"] == "assistant":
            violation = check_redline(m["content"])
            if violation:
                sample["_meta"]["redline_violation"] = violation
                return None

    return sample


def main():
    input_dir = Path("data/processed")
    output_path = Path("data/processed/cleaned_data_v1.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有输入文件
    input_files = []
    for d in ["public_data", "synthetic_data", "expert_data"]:
        dpath = input_dir / d
        if dpath.exists():
            input_files.extend(dpath.glob("*.jsonl"))
    # 只保留规范化的专家数据池（排除core/normal/filtered分开文件）
    input_files = [f for f in input_files if "expert_pool_v1_norm" in f.name or "expert_data" not in str(f)]

    logger.info(f"Input files: {[f.name for f in input_files]}")

    all_samples = []
    for f in input_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            all_samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    logger.info(f"Total raw samples: {len(all_samples)}")

    # 1. 格式/角色/语言/长度/红线 清洗
    cleaned = []
    stats = Counter()
    for s in all_samples:
        result = clean_sample(s)
        if result:
            cleaned.append(result)
        else:
            stats["filtered"] += 1

    logger.info(f"After basic cleaning: {len(cleaned)}")

    # 2. 精确去重
    before_dedup = len(cleaned)
    cleaned = exact_dedup(cleaned)
    logger.info(f"Exact dedup: {before_dedup} -> {len(cleaned)} ({before_dedup - len(cleaned)} removed)")

    # 3. MinHash 去重（如可用）
    before_minhash = len(cleaned)
    cleaned = minhash_dedup(cleaned)
    logger.info(f"MinHash dedup: {before_minhash} -> {len(cleaned)} ({before_minhash - len(cleaned)} removed)")

    # 统计来源分布
    source_dist = Counter(s["_meta"].get("source", "unknown") for s in cleaned)
    logger.info(f"Source distribution: {dict(source_dist)}")

    # 写入
    with open(output_path, "w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Cleaned data saved to {output_path} ({len(cleaned)} samples)")


if __name__ == "__main__":
    main()
