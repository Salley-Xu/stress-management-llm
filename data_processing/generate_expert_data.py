"""
专家数据生成 + Rubric 评分筛选脚本

流程:
  1. 用 DeepSeek 以"专家角色"生成高质量压力管理对话
  2. 覆盖步骤5设计的7类难点场景
  3. 用 DeepSeek 按专家Rubric 7维度(0-5) 自动评分
  4. 按综合评分筛选: >=28核心SFT, 24-28普通SFT, <24过滤
  5. 附加硬规则: 安全性<=2 或 共情=1 → 直接过滤

用法:
  # 生成1000条候选
  python data_processing/generate_expert_data.py --num_samples 1000
"""

import json
import os
import sys
import time
import random
import logging
import argparse
import requests
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

# ============================================================
# 7类难点场景（来自 expert_data_design.md）
# ============================================================
DIFFICULTY_SCENARIOS = [
    {"id": "E1", "name": "用户只想被倾听", "desc": "用户明确表示不需要建议，只想倾诉。助手应克制给建议，专注共情和倾听。"},
    {"id": "E2", "name": "多个压力源同时存在", "desc": "工作+家庭+健康等多领域压力交织，助手需协助理清优先级。"},
    {"id": "E3", "name": "用户信息不完整", "desc": "用户表达模糊、回避细节。助手需温和澄清而非假设。"},
    {"id": "E4", "name": "用户拒绝或质疑建议", "desc": "助手给出建议后被拒绝，需调整策略而非重复。"},
    {"id": "E5", "name": "已尝试方法但无效", "desc": "用户说'都试过了没用'，需探索为何无效并换思路。"},
    {"id": "E6", "name": "用户目标在多轮中改变", "desc": "从倾诉→澄清→规划→回访的完整旅程。"},
    {"id": "E7", "name": "安全边界和高风险表达", "desc": "隐晦风险信号、渐进升级、明确风险，需安全回应。"},
]

# 压力领域（对应 Rubric stress_type）
STRESS_TYPES = {
    "academic": "学习和考试压力",
    "work": "工作负荷和职场压力",
    "career": "求职和职业不确定性",
    "interpersonal": "人际关系和社交",
    "relationship": "恋爱和亲密关系",
    "family": "家庭和代际压力",
    "financial": "经济压力",
    "sleep": "睡眠和生活节律",
    "migration": "搬迁和环境适应",
    "procrastination": "长期任务和拖延",
}

# 用户意图（对应 Rubric user_intent）
USER_INTENTS = [
    "emotional_support",   # 情绪支持
    "problem_exploration", # 问题探索
    "advice_seeking",      # 建议寻求
    "action_planning",     # 行动规划
    "high_risk",           # 高风险支持
]

# 策略（对应 Rubric strategy）
STRATEGIES = [
    "empathy",        # 情绪支持/共情
    "exploration",    # 澄清探索
    "cognitive_reframe", # 认知调整
    "action_planning",   # 行动规划
    "risk_handling",     # 风险处理
]


def call_deepseek(prompt: str, max_tokens: int = 1024, api_key: str = None) -> str:
    """调用 DeepSeek API"""
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }
    resp = requests.post(API_URL, headers=headers, json=data, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ============================================================
# 1. 生成 prompt
# ============================================================
GENERATION_PROMPT = """你是一位受过专业训练的压力管理支持专家，擅长情绪支持和心理疏导。请根据以下场景要求，生成一段高质量的中文支持对话。

【难点类型】{difficulty}
【压力类型】{stress_type}（{stress_type_desc}）
【用户意图】{user_intent}
【使用策略】{strategy}
【严重度】{severity}

生成要求：
1. 用户的消息要真实自然，像真实的人在倾诉（有细节、有情绪、可能含糊其辞）
2. 助手的回复要体现专家级的支持技巧：
   - 共情具体贴切，认可用户的体验
   - 识别用户真实需求，不急于给建议
   - 策略使用与用户需求匹配，时机恰当
   - 建议具体、低门槛、可执行
   - 安全场景中正确识别风险并安全回应
3. 对话自然流畅，避免模板化表达
4. {multi_turn_note}

输出JSON格式：
{json_format}"""

# 单轮 JSON 示例
SINGLE_TURN_JSON = """{{
  "user": "用户的第一句话",
  "assistant": "助手的回应",
  "user_intent": "{user_intent}",
  "strategy": ["使用的策略"],
  "severity": "{severity}"
}}"""

# 多轮 JSON 示例（conversation 数组）
MULTI_TURN_JSON = """{{
  "conversation": [
    {{"user": "用户第一句话", "assistant": "助手回应", "strategy": ["empathy"]}},
    {{"user": "用户补充信息/状态变化", "assistant": "助手继续回应", "strategy": ["exploration"]}}
  ],
  "user_intent": "{user_intent}",
  "strategy": ["empathy", "exploration"],
  "severity": "{severity}"
}}"""


# ============================================================
# 2. 评分 prompt
# ============================================================
SCORING_PROMPT = """你是一位评审专家。请根据以下专家Rubric对一段压力管理对话的助手回复进行评分。

【用户消息】
{user_msg}

【助手回复】
{assistant_msg}

【评分维度】（每项0-5分）
1. 压力状态识别(0-5)：是否精准识别压力来源、情绪、严重度和潜在需求
2. 共情质量(0-5)：是否认可体验、避免否定、建立安全感
3. 用户需求识别(0-5)：是否准确判断用户需要倾诉/澄清/建议/规划/高风险支持
4. 策略选择(0-5)：策略是否与需求匹配、时机是否恰当
5. 建议可执行性(0-5)：建议是否具体、低门槛、符合用户状态
6. 安全性(0-5)：是否避免诊断、越界建议、风险处理是否正确
7. 对话自然性(0-5)：是否自然、避免模板化

输出JSON格式：
{{
  "stress_identification": 分数,
  "empathy": 分数,
  "need_identification": 分数,
  "strategy_selection": 分数,
  "actionability": 分数,
  "safety": 分数,
  "naturalness": 分数,
  "total": 总分(7项之和,0-35),
  "category": "core|normal|filter"
}}"""


def build_generation_prompt(scenario: dict, stress_type: str, intent: str, strategy: str, multi_turn: bool) -> str:
    severity = random.choice(["轻度", "中度", "持续较重"])
    if scenario["id"] == "E7":
        severity = "出现风险信号（L1-L2）"
    if multi_turn:
        multi_turn_note = "请生成2-3轮对话（含信息补充或状态变化），使用conversation数组字段。"
        json_format = MULTI_TURN_JSON.format(user_intent=intent, severity=severity)
    else:
        multi_turn_note = "请生成1轮对话（user + assistant），使用user/assistant字段。"
        json_format = SINGLE_TURN_JSON.format(user_intent=intent, severity=severity)
    return GENERATION_PROMPT.format(
        difficulty=f"{scenario['id']} - {scenario['name']}：{scenario['desc']}",
        stress_type=stress_type,
        stress_type_desc=STRESS_TYPES[stress_type],
        user_intent=intent,
        strategy=strategy,
        severity=severity,
        multi_turn_note=multi_turn_note,
        json_format=json_format,
    )


def build_scoring_prompt(user_msg: str, assistant_msg: str) -> str:
    return SCORING_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)


def parse_json(text: str) -> dict:
    """解析JSON输出（含容错）"""
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


def generate_one(scenario: dict, api_key: str) -> dict:
    """生成一条对话 + 评分"""
    stress_type = random.choice(list(STRESS_TYPES.keys()))
    intent = random.choice(USER_INTENTS)
    strategy = random.choice(STRATEGIES)
    multi_turn = scenario["id"] in ("E4", "E5", "E6", "E7")

    # 步骤1: 生成（多轮对话较长，需2048 tokens避免截断；失败重试）
    gen_prompt = build_generation_prompt(scenario, stress_type, intent, strategy, multi_turn)
    gen_data = {}
    for attempt in range(2):
        try:
            gen_text = call_deepseek(gen_prompt, max_tokens=2048, api_key=api_key)
            gen_data = parse_json(gen_text)
            if gen_data:
                break
        except Exception:
            pass
    if not gen_data:
        return {"status": "error", "error": "generation failed"}

    # 支持多轮（conversation字段）
    user_msg = gen_data.get("user", "")
    assistant_msg = gen_data.get("assistant", "")
    if "conversation" in gen_data and isinstance(gen_data["conversation"], list):
        # 多轮：提取首轮user和末轮assistant
        user_msgs = []
        assistant_msgs = []
        for turn in gen_data["conversation"]:
            if isinstance(turn, dict):
                for k, v in turn.items():
                    kk = k.strip().lower()
                    if kk in ("user", "用户", "用户消息", "seeker"):
                        user_msgs.append(str(v))
                    elif kk in ("assistant", "助手", "助手回复", "回复", "回答", "supporter"):
                        assistant_msgs.append(str(v))
        if user_msgs:
            user_msg = user_msgs[0]
        if assistant_msgs:
            assistant_msg = assistant_msgs[-1]

    if not user_msg or not assistant_msg:
        return {"status": "error", "error": "missing content"}

    # 步骤2: 评分（失败重试2次）
    score_prompt = build_scoring_prompt(user_msg, assistant_msg)
    score_data = {}
    for attempt in range(3):
        try:
            score_text = call_deepseek(score_prompt, max_tokens=1024, api_key=api_key)
            score_data = parse_json(score_text)
            if score_data and "total" in score_data:
                break
        except Exception:
            pass
    if not score_data or "total" not in score_data:
        return {"status": "error", "error": "scoring failed"}

    return {
        "status": "ok",
        "user_msg": user_msg,
        "assistant_msg": assistant_msg,
        "stress_type": stress_type,
        "user_intent": gen_data.get("user_intent", intent),
        "strategy": gen_data.get("strategy", [strategy]),
        "severity": gen_data.get("severity", ""),
        "difficulty_type": scenario["id"],
        "rubric_score": score_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/processed/expert_data")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("No DEEPSEEK_API_KEY. Set env or .env")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出文件: 核心/普通分开
    core_path = output_dir / f"expert_core_{ts}.jsonl"
    normal_path = output_dir / f"expert_normal_{ts}.jsonl"
    filtered_path = output_dir / f"expert_filtered_{ts}.jsonl"

    core_f = open(core_path, "w", encoding="utf-8")
    normal_f = open(normal_path, "w", encoding="utf-8")
    filtered_f = open(filtered_path, "w", encoding="utf-8")

    stats = {"generated": 0, "core": 0, "normal": 0, "filtered": 0, "error": 0}
    t0 = time.time()
    max_attempts = args.num_samples * 3  # 最多尝试3倍，防止无限循环

    logger.info(f"Target: {args.num_samples} candidates, 7 difficulty types, max_attempts={max_attempts}")

    attempts = 0
    while stats["generated"] < args.num_samples and attempts < max_attempts:
        attempts += 1
        scenario = random.choice(DIFFICULTY_SCENARIOS)
        result = generate_one(scenario, api_key)

        if result["status"] == "error":
            stats["error"] += 1
            continue

        score = result.get("rubric_score", {})
        total = score.get("total", 0)
        safety = score.get("safety", 5)
        empathy = score.get("empathy", 5)

        # 硬规则过滤
        if safety <= 2 or empathy == 1 or total < 24:
            category = "filter"
        elif total >= 28:
            category = "core"
        else:
            category = "normal"

        # 构建记录
        record = {
            "_meta": {
                "id": f"expert_{ts}_{stats['generated']:05d}",
                "source": "synthetic",
                "source_name": "DeepSeek v4-flash (expert role)",
                "generation_method": "llm_synthetic",
                "review_status": "pending",
                "version": "1.0",
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "usage": "train",
                "language": "zh-CN",
                "rubric_category": category,
            },
            "type": "sft_single",
            "labels": {
                "domains": [result["stress_type"]],
                "severity": "SV-MOD",
                "user_goals": [result["user_intent"]],
                "strategies": result["strategy"],
                "difficulty_type": result["difficulty_type"],
            },
            "messages": [
                {"role": "user", "content": result["user_msg"]},
                {"role": "assistant", "content": result["assistant_msg"]},
            ],
            "rubric_score": score,
        }

        if category == "core":
            core_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["core"] += 1
        elif category == "normal":
            normal_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["normal"] += 1
        else:
            filtered_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["filtered"] += 1

        stats["generated"] += 1

        if stats["generated"] % 20 == 0:
            elapsed = time.time() - t0
            rate = stats["generated"] / elapsed if elapsed > 0 else 0
            logger.info(
                f"  {stats['generated']}/{args.num_samples} "
                f"(core={stats['core']} normal={stats['normal']} filtered={stats['filtered']} err={stats['error']} {rate:.1f}/s)"
            )

    core_f.close(); normal_f.close(); filtered_f.close()

    if attempts >= max_attempts:
        logger.warning(f"Reached max_attempts={max_attempts} before target. ")

    logger.info("=" * 50)
    logger.info(f"Done. core={stats['core']}, normal={stats['normal']}, filtered={stats['filtered']}, error={stats['error']}")
    logger.info(f"Core: {core_path}")
    logger.info(f"Normal: {normal_path}")
    logger.info(f"Filtered: {filtered_path}")


if __name__ == "__main__":
    main()
