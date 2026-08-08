"""
SFT-Stage2 训练数据生成脚本：多轮训练集V2 + 安全专项训练集V2

针对步骤9评测暴露的问题（风险漏检↑、回复过短、压力源理解错误↑），
按步骤10要求补充两类数据：
  1. 多轮训练集V2：6类多轮场景（M1-M6），训练前文保持/策略调整/状态感知
  2. 安全专项训练集V2：6类安全专项场景（S1-S6），训练安全边界回应

用法:
  # 生成多轮数据800条
  python data_processing/generate_sft2_data.py --mode multi --num_samples 800
  # 生成安全专项数据500条
  python data_processing/generate_sft2_data.py --mode safety --num_samples 500
"""

import json
import os
import re
import time
import random
import logging
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"

load_dotenv()

# ============================================================
# 场景定义（步骤10）
# ============================================================
# 多轮场景：核心行为 = 前文信息保持、避免重复无效建议、根据反馈调整策略、状态感知
MULTI_TURN_SCENARIOS = [
    {"id": "M1", "name": "用户逐步补充信息", "desc": "用户一开始表达模糊，后续轮次逐步补充关键信息（背景/细节/真实困扰）。助手需根据新信息调整理解，不固化早期判断。"},
    {"id": "M2", "name": "用户拒绝建议", "desc": "用户明确拒绝或质疑助手的建议。助手不能重复原建议，需转向共情、澄清用户顾虑，再换策略。"},
    {"id": "M3", "name": "已尝试方法无效", "desc": "用户说'都试过了/没用'。助手需探索为何无效、检查执行障碍，换一个方向而非重复。"},
    {"id": "M4", "name": "用户目标发生变化", "desc": "用户从倾诉→澄清→规划→调整目标，需求随轮次改变。助手需跟随目标变化调整支持方式。"},
    {"id": "M5", "name": "压力状态改善或恶化", "desc": "用户压力状态在轮次间明显好转或恶化。助手需感知变化并调整语气（好转时肯定进展，恶化时加强支持）。"},
    {"id": "M6", "name": "早期普通、后期出现风险信号", "desc": "对话前期是普通压力，后期用户表达出现风险信号（绝望、自伤念头等）。助手需及时识别风险并引导专业帮助，而非继续普通问题解决式建议。"},
]

# 安全专项场景：核心行为 = 不诊断/不荐药/不保证/不保密(涉风险)/不诱导依赖/识别隐晦风险
SAFETY_SCENARIOS = [
    {"id": "S1", "name": "诊断请求", "desc": "用户请求模型诊断自己是否有某种心理疾病。助手不能给出诊断，应温和说明自己非临床评估，建议专业评估。"},
    {"id": "S2", "name": "药物请求", "desc": "用户询问或要求推荐药物/剂量。助手不能推荐药物，应建议咨询医生或精神科专业人员。"},
    {"id": "S3", "name": "诱导模型保证结果", "desc": "用户要求模型保证'会好起来/一定能解决/不会有问题'。助手不能做出绝对保证，应诚实表达不确定性并给予支持。"},
    {"id": "S4", "name": "诱导保密", "desc": "用户要求助手承诺'替你保密，不告诉任何人'，尤其涉及自伤风险。助手应说明涉及人身安全时不能承诺保密，鼓励寻求帮助。"},
    {"id": "S5", "name": "诱导依赖", "desc": "用户表达'只有你能帮我/我离不开你/我只想跟你聊'。助手不应鼓励依赖，应温和鼓励现实支持系统和自主能力。"},
    {"id": "S6", "name": "隐晦或渐进式高风险表达", "desc": "用户用隐晦/渐进方式表达风险（'没意思''想消失''最近总梦到出事'）。助手需识别信号、直接但温和地关心，并引导专业帮助。"},
]

# 合法领域标签（DS-*）与描述
DOMAIN_DESCS = {
    "DS-LRN": "学习和考试压力",
    "DS-WRK": "工作负荷和职场压力",
    "DS-CAR": "求职和职业不确定性",
    "DS-INT": "人际关系和社交",
    "DS-REL": "恋爱和亲密关系",
    "DS-FAM": "家庭和代际压力",
    "DS-FIN": "经济压力",
    "DS-SLP": "睡眠和生活节律",
    "DS-MIG": "搬迁和环境适应",
    "DS-PRC": "长期任务和拖延",
}
VALID_DOMAINS = list(DOMAIN_DESCS.keys())

# 用户目标标签（UG-*）
USER_GOALS = [
    "UG-VEN",  # 倾诉
    "UG-CLA",  # 澄清/被理解
    "UG-ADV",  # 寻求建议
    "UG-PLN",  # 行动规划
    "UG-SEC",  # 安全感/支持
]


def call_deepseek(prompt: str, max_tokens: int = 4096, api_key: str = None) -> str:
    """调用 DeepSeek API，带网络重试（4次+退避）
    注：deepseek-v4-flash 带 reasoning，reasoning_tokens 会占输出空间导致 content 偶发为空，
    因此 max_tokens 需给足（4096），且对空响应做重试。
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=180)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                # reasoning 占满 max_tokens 导致 content 为空 → 立即重试
                raise RuntimeError("empty content (reasoning consumed budget)")
            return content
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt * 2)  # 2,4,8 秒退避
    raise RuntimeError(f"API call failed after 4 retries: {last_err}")


def parse_json(text: str) -> dict:
    """解析JSON输出（含容错）"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ============================================================
# 多轮场景 prompt
# ============================================================
MULTI_TURN_PROMPT = """你是一位受过专业训练的压力管理支持专家。请根据下面的场景要求，生成一段{min_turns}-{max_turns}轮的中文多轮支持对话（每轮=1条用户消息+1条助手回复，共{min_turns}-{max_turns}组）。

【场景】{scenario_id} - {scenario_name}：{scenario_desc}
【压力领域】{domain}（{domain_desc}）

生成要求：
1. 用户消息要真实自然，像真实的人在倾诉——有细节、有情绪、随轮次逐渐变化
2. 对话必须围绕该场景的核心行为展开（见场景描述），且核心行为主要体现在【最后一条助手回复】中
3. 助手每一轮都要体现专业支持技巧：共情贴切、不抢先建议、建议具体低门槛
4. 最后一轮用户消息必须是一个"需要助手基于前文信息作出回应"的提问或倾诉
5. 对话自然流畅，避免模板化，不要用"我理解你的感受"这类空泛套话
6. 轮数不少于{min_turns}轮

输出JSON格式：
{{
  "conversation": [
    {{"user": "用户第1条", "assistant": "助手第1条回复"}},
    {{"user": "用户第2条", "assistant": "助手第2条回复"}},
    {{"user": "用户第3条", "assistant": "助手第3条回复"}}
  ],
  "domain": "{domain}",
  "user_goal": "UG-VEN 或其他UG-*标签",
  "severity": "SV-MOD/SV-IMP/SV-RISK等"
}}"""


# ============================================================
# 安全专项 prompt
# ============================================================
SAFETY_PROMPT = """你是一位受过专业训练的压力管理支持专家。请根据下面的安全边界场景，生成一段2-4轮的中文多轮对话（每轮=1条用户消息+1条助手回复）。

【安全场景】{scenario_id} - {scenario_name}：{scenario_desc}

生成要求：
1. 用户消息要自然真实，像真实的人在倾诉或提问，不要直接暴露"这是测试"
2. 对话核心看【助手回复】：助手必须正确处理安全边界，体现以下行为：
   - 不给出临床诊断、不推荐药物或剂量、不承诺/保证结果
   - 对用户表达关心、认可其求助的勇气
   - 明确指出自己能力的边界（如"我无法做专业评估"），并【建议用户寻求专业帮助】（心理科/精神科/咨询师/热线）
   - 涉及自伤风险时不承诺保密，鼓励联系可信任的人或专业机构
   - 对隐晦风险信号要温和直接地询问澄清，不回避
3. 助手回复要自然温暖，不能机械拒答、不能冷冰冰
4. 至少2轮对话，让用户有追问，助手在后续轮次继续正确引导

输出JSON格式：
{{
  "conversation": [
    {{"user": "用户第1条", "assistant": "助手第1条回复"}},
    {{"user": "用户第2条", "assistant": "助手第2条回复"}}
  ],
  "domain": "DS-INT 或其他DS-*标签",
  "user_goal": "UG-***",
  "severity": "SV-RISK或对应严重度"
}}"""


def build_multi_prompt(scenario: dict) -> str:
    domain = random.choice(VALID_DOMAINS)
    min_turns = random.choice([3, 4])
    max_turns = min_turns + 1
    return MULTI_TURN_PROMPT.format(
        scenario_id=scenario["id"],
        scenario_name=scenario["name"],
        scenario_desc=scenario["desc"],
        domain=domain,
        domain_desc=DOMAIN_DESCS[domain],
        min_turns=min_turns,
        max_turns=max_turns,
    )


def build_safety_prompt(scenario: dict) -> str:
    domain = random.choice(VALID_DOMAINS)
    return SAFETY_PROMPT.format(
        scenario_id=scenario["id"],
        scenario_name=scenario["name"],
        scenario_desc=scenario["desc"],
        domain=domain,
    )


def extract_conversation(data: dict) -> list:
    """从生成结果提取 conversation → [{"role","content"}, ...] 交替消息列表"""
    conv = data.get("conversation") or data.get("dialogue") or data.get("messages")
    if not isinstance(conv, list):
        return []
    messages = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        user_text = None
        asst_text = None
        for k, v in turn.items():
            kk = str(k).strip().lower()
            if kk in ("user", "用户", "用户消息", "seeker", "来访者"):
                user_text = str(v).strip()
            elif kk in ("assistant", "助手", "助手回复", "回复", "回答", "supporter", "咨询师"):
                asst_text = str(v).strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    return messages


def valid_messages(messages: list, min_turns: int = 3) -> bool:
    """校验消息序列：role交替、轮数足够、最后一条是assistant、内容非空"""
    if len(messages) < min_turns * 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    # role必须从user开始且交替
    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return False
        if len(m["content"]) < 8:
            return False
    return True


# 安全专项：助手回复必须体现的引导信号（粗筛）
SAFETY_SIGNALS = [
    "专业", "医生", "咨询", "心理科", "精神科", "热线", "机构", "求助", "寻求帮助",
    "支持", "评估", "中心", "门诊", "建议你", "建议您", "建议可以", "联系",
]


def safety_check(messages: list) -> bool:
    """安全专项粗筛：最后一条assistant回复需包含引导专业帮助的信号"""
    last = messages[-1]["content"]
    hits = sum(1 for s in SAFETY_SIGNALS if s in last)
    return hits >= 2


def generate_one(scenario: dict, mode: str, api_key: str) -> dict:
    """生成一条数据"""
    if mode == "multi":
        prompt = build_multi_prompt(scenario)
    else:
        prompt = build_safety_prompt(scenario)

    gen_data = {}
    for attempt in range(3):
        try:
            text = call_deepseek(prompt, max_tokens=4096, api_key=api_key)
            gen_data = parse_json(text)
            if gen_data:
                break
        except Exception:
            time.sleep(1)
    if not gen_data:
        return {"status": "error", "error": "generation failed"}

    messages = extract_conversation(gen_data)
    if not valid_messages(messages, min_turns=2):
        return {"status": "error", "error": "invalid conversation"}

    if mode == "safety" and not safety_check(messages):
        return {"status": "error", "error": "safety signal missing"}

    domain = str(gen_data.get("domain", "")).strip().upper()
    if domain not in VALID_DOMAINS:
        domain = random.choice(VALID_DOMAINS)
    user_goal = str(gen_data.get("user_goal", "")).strip().upper()
    if not user_goal.startswith("UG-") or len(user_goal) != 5:
        user_goal = "UG-VEN"
    severity = str(gen_data.get("severity", "")).strip().upper()
    if not severity.startswith("SV-"):
        severity = "SV-MOD" if scenario["id"] in ("M1", "M2", "M3", "M4", "M5") else "SV-RISK"

    return {
        "status": "ok",
        "messages": messages,
        "domain": domain,
        "user_goal": user_goal,
        "severity": severity,
        "scenario_id": scenario["id"],
    }


def build_record(result, label_type, prefix, ts, seq, worker_idx=None):
    return {
        "_meta": {
            "id": f"sft2_{prefix}_{ts}_{seq:05d}",
            "source": "synthetic",
            "source_name": "DeepSeek v4-flash",
            "generation_method": "llm_synthetic",
            "review_status": "pending",
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "usage": "train",
            "language": "zh-CN",
            "stage2_scenario": result["scenario_id"],
            "stage2_worker": worker_idx,
        },
        "type": label_type,
        "labels": {
            "domains": [result["domain"]],
            "severity": result["severity"],
            "user_goals": [result["user_goal"]],
            "strategies": [],
            "difficulty_type": result["scenario_id"],
        },
        "messages": result["messages"],
    }


def worker_main(mode, scenarios, label_type, prefix, ts, worker_idx, num_workers, num_samples, out_file, api_key):
    """单worker循环生成，写入独立文件，末尾关闭"""
    target = num_samples // num_workers + (1 if worker_idx < num_samples % num_workers else 0)
    stats = {"ok": 0, "error": 0}
    t0 = time.time()
    try:
        if target <= 0:
            return stats
        while stats["ok"] < target:
            scenario = random.choice(scenarios)
            result = generate_one(scenario, mode, api_key)
            if result["status"] == "error":
                stats["error"] += 1
                continue
            seq = worker_idx * 100000 + stats["ok"]
            record = build_record(result, label_type, prefix, ts, seq, worker_idx)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 10 == 0:
                elapsed = time.time() - t0
                rate = stats["ok"] / elapsed if elapsed > 0 else 0
                logger.info(f"  [w{worker_idx}] {stats['ok']}/{target} (err={stats['error']} {rate:.1f}/s)")
    finally:
        out_file.close()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["multi", "safety"], required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data/processed/sft2_data")
    parser.add_argument("--workers", type=int, default=3, help="并发worker数")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("No DEEPSEEK_API_KEY. Set .env or env")
        return

    if args.mode == "multi":
        scenarios = MULTI_TURN_SCENARIOS
        prefix = "multi_turn"
        label_type = "sft_multi"
    else:
        scenarios = SAFETY_SCENARIOS
        prefix = "safety"
        label_type = "sft_safety"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{prefix}_{ts}.jsonl"

    tmp_dir = output_dir / f"tmp_{prefix}_{ts}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target: {args.num_samples} ({args.mode}), workers={args.workers}")
    t0 = time.time()
    total_stats = {"ok": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i in range(args.workers):
            f = open(tmp_dir / f"worker_{i}.jsonl", "w", encoding="utf-8")
            futures.append(ex.submit(
                worker_main, args.mode, scenarios, label_type, prefix, ts,
                i, args.workers, args.num_samples, f, api_key,
            ))
        for fut in futures:
            st = fut.result()
            total_stats["ok"] += st["ok"]
            total_stats["error"] += st["error"]

    # 合并worker文件
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(args.workers):
            wp = tmp_dir / f"worker_{i}.jsonl"
            if wp.exists():
                for line in open(wp, "r", encoding="utf-8"):
                    out.write(line)
                wp.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    logger.info("=" * 50)
    logger.info(f"Done. ok={total_stats['ok']}, error={total_stats['error']}, time={time.time()-t0:.0f}s")
    logger.info(f"Output: {out_path}")


if __name__ == "__main__":
    main()
