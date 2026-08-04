"""
LLM 合成数据生成脚本

支持两种模式:
  1. API 模式: 调用 DeepSeek/GPT API（推荐）
  2. Local 模式: 使用本地模型（概念验证用）

用法:
  # API模式 (DeepSeek)
  python data_processing/generate_synthetic.py \
      --mode api --provider deepseek \
      --api_key $DEEPSEEK_API_KEY \
      --num_samples 100

  # Local模式 (概念验证)
  python data_processing/generate_synthetic.py \
      --mode local --model_path "D:/anaconda3/envs/stress-mgmt/models/.../Qwen2.5-7B-Instruct/.../master" \
      --num_samples 10
"""

import json
import os
import sys
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = (
    "你是一个提供日常压力管理支持的中文助手。你的任务是倾听和理解用户的感受，"
    "帮助梳理压力来源，在需要时一起制定可行的小步骤，并在适当情况下建议用户寻求专业帮助。"
    "\n\n重要原则：\n"
    "- 先倾听和理解，再给建议\n"
    "- 不抢先给建议，不一次给太多任务\n"
    "- 保留用户的选择权，不命令用户\n"
    "- 不假设你未获得的信息\n"
    "- 你不做临床诊断，不提供治疗或药物建议\n"
    "- 对于明显高风险的情况，使用谨慎、支持性的表达，鼓励用户寻求专业支持"
)

# ============================================================
# 人物画像库
# ============================================================
PERSONAS = [
    {"id": "student", "age": "21岁", "role": "大学生", "context": "大三在读，课业繁重，面临考研/就业选择"},
    {"id": "new_grad", "age": "24岁", "role": "职场新人", "context": "刚工作一年，在陌生城市独自租房，社交圈小"},
    {"id": "mid_career", "age": "35岁", "role": "中层管理", "context": "已婚有小孩，工作压力大，上有老下有小"},
    {"id": "parent", "age": "30岁", "role": "全职父母", "context": "在家带孩子，经济依赖配偶，缺乏个人空间"},
    {"id": "freelancer", "age": "29岁", "role": "自由职业者", "context": "收入不稳定，没有同事社交，作息不规律"},
    {"id": "retired", "age": "62岁", "role": "退休人员", "context": "刚退休，子女在外地，感到生活失去重心"},
]

# ============================================================
# Prompt 模板库
# ============================================================

SFT_SINGLE_TEMPLATES = [
    # 模板1: 直接生成
    """你是一位擅长情绪支持对话的AI。请为以下场景生成一个高质量的对话。

【用户画像】{persona}
【压力场景】{domain_desc}
【压力程度】{severity_desc}
【用户目标】{goal_desc}
【用户表达风格】{style_desc}
【要求策略】{strategy_desc}

请生成一个自然的中文对话（一轮即可）：
1. 用户的消息应该真实、自然，符合人物画像和压力场景
2. 助手的回复必须使用要求的策略，且语言自然不做作
3. 助手回复{extra_constraint}

输出格式（JSON）:
{{"user": "用户消息", "assistant": "助手回复"}}
""",

    # 模板2: 先给反例再生成
    """你是一位情绪支持对话专家。请生成一段高质量对话。

场景: {domain_desc}
用户: {persona}
感受: {severity_desc}
需要: {goal_desc}
风格: {style_desc}

注意事项:
- 用户的表达应该{style_desc}
- 助手应该{strategy_desc}
- {extra_constraint}
- 避免: 抢先给建议、空泛共情、一次给太多建议

输出JSON: {{"user": "...", "assistant": "..."}}
""",

    # 模板3: 详细引导
    """请扮演一位需要压力管理支持的用户，并生成助手的回复。

用户背景: {persona}
当前处境: {domain_desc}，{severity_desc}
用户来对话的目的是: {goal_desc}
表达方式: {style_desc}

助手需要:
- {strategy_desc}
- {extra_constraint}

先写用户的话（1-3句话），再写助手的回复。输出JSON格式。
""",
]

MULTITURN_TEMPLATES = [
    """生成一个多轮压力管理对话（{num_turns}轮）。

用户: {persona}，面临{domain_desc}，{severity_desc}
对话弧线: 用户最初{goal_start}，随着对话推进逐渐{goal_end}

每轮要求:
{round_requirements}

输出JSON:
{{"conversation": [{{"turn": 1, "user": "...", "assistant": "...", "strategies": [...]}}, ...]}}
""",
]

SAFETY_TEMPLATES = [
    """生成一个安全边界场景的对话。

风险类型: {risk_type}
风险等级: {risk_level}
用户: {persona}

要求:
1. 用户的表达{explicitness}包含风险信号
2. 助手必须识别风险并做安全回应（使用ST-SAF策略）
3. 助手不能: 做诊断、推荐药物、弱化风险、承诺保密
4. 助手回复要温和但坚定

输出JSON: {{"user": "...", "assistant": "...", "risk_type": "{risk_type}"}}
""",
]


def build_domain_desc(domain: str) -> str:
    mapping = {
        "DS-LRN": "学习和考试压力，复习备考、论文压力、成绩焦虑",
        "DS-WRK": "工作负荷大，加班多，项目压力，职场人际关系",
        "DS-CAR": "求职面试不顺，职业发展方向迷茫",
        "DS-INT": "人际关系紧张，社交孤独，与朋友/同事矛盾",
        "DS-REL": "恋爱/婚姻关系出现问题，感情困扰",
        "DS-FAM": "家庭矛盾，父母期望压力，催婚催生",
        "DS-FIN": "经济压力，房贷/房租负担，收入焦虑",
        "DS-SLP": "失眠、作息紊乱、慢性疲劳",
        "DS-MIG": "到新城市/国家适应困难，没有社交圈",
        "DS-PRC": "长期拖延导致任务堆积，完美主义导致的行动瘫痪",
    }
    return mapping.get(domain, domain)


def build_severity_desc(severity: str) -> str:
    mapping = {
        "SV-MLD": "轻度压力，暂时性的，生活功能基本正常",
        "SV-MOD": "压力明显，已影响心情和睡眠，但还能应付日常",
        "SV-PER": "压力持续数周，影响多个生活领域，感到疲惫不堪",
        "SV-IMP": "压力已严重影响工作/学习/社交，想要逃避",
        "SV-RSK": "出现绝望感或'活着没意思'的想法（不要直接写自杀，用隐晦表达）",
    }
    return mapping.get(severity, severity)


def build_goal_desc(goal: str) -> str:
    mapping = {
        "UG-HRD": "希望被理解和倾听，不想要建议，只想倾诉",
        "UG-CLA": "思绪混乱，希望帮助理清问题和感受",
        "UG-STB": "当前情绪强烈（焦虑/难过/愤怒），需要先稳定下来",
        "UG-DEC": "面临选择困境，需要分析利弊帮助做决定",
        "UG-PLN": "已理清问题，想要具体的行动计划",
        "UG-REV": "之前尝试了一些方法，回来汇报进展并调整",
    }
    return mapping.get(goal, goal)


def build_strategy_desc(strategies: List[str]) -> str:
    mapping = {
        "ST-RFL": "用反映式倾听回应用户的情绪（如'听起来你...'）",
        "ST-VAL": "确认和接纳用户的感受（如'这种感觉是很正常的'）",
        "ST-OPN": "用开放式问题帮助用户深入表达",
        "ST-SUM": "总结复述用户表达的内容，确保理解正确",
        "ST-DEC": "帮助用户将模糊压力拆解为具体可管理的部分",
        "ST-REF": "引导用户从不同角度看待问题（认知重评）",
        "ST-PRB": "协作式问题解决：分析→头脑风暴→评估",
        "ST-MIC": "制定微小、具体、可执行的下一步行动",
        "ST-STB": "提供即时的情绪稳定技巧（如呼吸、grounding）",
        "ST-SOC": "引导用户识别和激活已有的社会支持",
        "ST-SAF": "温和地引导寻求专业心理帮助",
    }
    descs = [mapping.get(s, s) for s in strategies]
    return "，".join(descs)


def generate_sample_local(
    model, tokenizer, prompt: str, max_new_tokens: int = 512
) -> str:
    """使用本地模型生成"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()


def generate_sample_api(provider: str, api_key: str, prompt: str) -> str:
    """使用API生成（DeepSeek/OpenAI兼容接口）"""
    import requests

    if provider == "deepseek":
        url = "https://api.deepseek.com/v1/chat/completions"
        model_name = "deepseek-v4-flash"
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        model_name = "gpt-4o"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 512,
    }

    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取JSON，兼容中文键"""
    # Try to find JSON block
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    result = None
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if not result or not isinstance(result, dict):
        return None

    # Normalize Chinese/alternate keys to user/assistant
    normalized = {}
    for k, v in result.items():
        kl = k.lower()
        if kl in ("用户", "user", "用户消息", "问题"):
            normalized["user"] = v
        elif kl in ("助手", "assistant", "回复", "助手回复", "回答"):
            normalized["assistant"] = v
        elif kl == "conversation":
            normalized["conversation"] = v
        else:
            normalized[k] = v

    return normalized


def build_sample_metadata(
    domain: str, severity: str, goal: str, persona: dict,
    strategies: List[str], template_id: int, mode: str
) -> dict:
    return {
        "_meta": {
            "id": f"syn_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            "source": "synthetic",
            "source_name": f"local-7b" if mode == "local" else "api",
            "generation_method": "llm_synthetic",
            "generation_config": {
                "teacher_model": "local" if mode == "local" else "api",
                "temperature": 0.8,
                "top_p": 0.9,
                "prompt_template_id": f"TEMPLATE-SFT-{template_id:03d}",
            },
            "combination_key": f"{domain}_{severity}_{goal}_{persona['id']}",
            "review_status": "pending",
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "usage": "train",
            "language": "zh-CN",
        },
        "type": "sft_single",
        "labels": {
            "domains": [domain],
            "severity": severity,
            "user_goals": [goal],
            "strategies": strategies,
        },
    }


def run_synthesis(args):
    """主合成流程"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"synthetic_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # 加载本地模型（仅local模式）
    model = None
    tokenizer = None
    if args.mode == "local" and args.model_path:
        logger.info(f"Loading local model: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use 4-bit quantization for 8GB GPUs
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.eval()
        logger.info("Model loaded (4-bit).")

    # 采样组合
    domains = ["DS-WRK", "DS-LRN", "DS-INT", "DS-REL", "DS-FAM", "DS-CAR", "DS-SLP"]
    severities = ["SV-MLD", "SV-MOD", "SV-PER"]
    goals = ["UG-HRD", "UG-CLA", "UG-PLN"]
    strategies_pool = [
        ["ST-RFL", "ST-VAL", "ST-OPN"],  # 倾听型
        ["ST-RFL", "ST-VAL", "ST-DEC"],  # 澄清型
        ["ST-MIC", "ST-SOC"],             # 行动型
    ]

    count = 0
    failed = 0
    target = args.num_samples

    logger.info(f"Target: {target} samples, mode: {args.mode}")
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        while count < target:
            # 随机采样组合
            domain = random.choice(domains)
            severity = random.choice(severities)
            goal = random.choice(goals)
            persona = random.choice(PERSONAS)
            strategies = random.choice(strategies_pool)
            template_id = random.randint(1, len(SFT_SINGLE_TEMPLATES))
            template = SFT_SINGLE_TEMPLATES[template_id - 1]

            # 构建 prompt
            extra = "不要一次给太多建议，保持对话的自然流动"
            prompt = template.format(
                persona=f"{persona['role']}，{persona['age']}，{persona['context']}",
                domain_desc=build_domain_desc(domain),
                severity_desc=build_severity_desc(severity),
                goal_desc=build_goal_desc(goal),
                style_desc=random.choice(["直接坦率", "犹豫不决", "情绪化", "尽量保持理性但藏不住情绪"]),
                strategy_desc=build_strategy_desc(strategies),
                extra_constraint=extra,
            )

            try:
                if args.mode == "local":
                    response = generate_sample_local(model, tokenizer, prompt)
                else:
                    response = generate_sample_api(args.provider, args.api_key, prompt)

                data = extract_json(response)
                if data and "user" in data and "assistant" in data:
                    # 构建标准schema
                    record = build_sample_metadata(domain, severity, goal, persona, strategies, template_id, args.mode)
                    record["messages"] = [
                        {"role": "user", "content": data["user"]},
                        {"role": "assistant", "content": data["assistant"]},
                    ]
                    record["quality_labels"] = {"errors": [], "notes": "Auto-generated, needs review"}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if count % 10 == 0:
                        elapsed = time.time() - t0
                        rate = count / elapsed if elapsed > 0 else 0
                        logger.info(f"  Progress: {count}/{target} ({rate:.1f}/s)")
                else:
                    failed += 1
                    if failed <= 3:
                        logger.warning(f"  JSON parse failed. Raw: {response[:200]}")

            except Exception as e:
                failed += 1
                if failed <= 3:
                    logger.warning(f"  Generation failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"Done: {count} generated, {failed} failed, {elapsed:.0f}s")
    logger.info(f"Output: {output_path}")

    # Print stats
    with open(output_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    domains_gen = {}
    for s in samples:
        d = s["labels"]["domains"][0]
        domains_gen[d] = domains_gen.get(d, 0) + 1
    logger.info("Domain distribution:")
    for d, c in sorted(domains_gen.items()):
        logger.info(f"  {d}: {c}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generation")
    parser.add_argument("--mode", choices=["api", "local"], default="local")
    parser.add_argument("--provider", choices=["deepseek", "openai"], default="deepseek")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="data/processed/synthetic_data")
    args = parser.parse_args()

    if args.mode == "api" and not args.api_key:
        args.api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not args.api_key:
            logger.error("API mode requires --api_key or DEEPSEEK_API_KEY/OPENAI_API_KEY env var")
            sys.exit(1)

    if args.mode == "local" and not args.model_path:
        args.model_path = "D:/anaconda3/envs/stress-mgmt/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
        logger.info(f"Using default model path: {args.model_path}")

    run_synthesis(args)


if __name__ == "__main__":
    main()
