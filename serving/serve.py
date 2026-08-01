"""
vLLM/SGLang 推理服务脚本

用法:
    # vLLM (推荐)
    python serving/serve.py --engine vllm \
        --model_path checkpoints/sm-sft-qwen-xxx/final_adapter \
        --port 8000

    # SGLang
    python serving/serve.py --engine sglang \
        --model_path checkpoints/sm-sft-qwen-xxx/final_adapter \
        --port 30000

    # Transformers 本地测试
    python serving/serve.py --engine transformers \
        --model_path checkpoints/sm-sft-qwen-xxx/final_adapter \
        --interactive
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def serve_vllm(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 64,
    quantization: Optional[str] = None,
):
    """使用 vLLM 启动 OpenAI 兼容 API 服务"""
    try:
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.api_server import main as vllm_server
    except ImportError:
        logger.error("vLLM 未安装，请运行: pip install vllm")
        return

    logger.info(f"启动 vLLM 服务: model={model_path}, port={port}")

    # vLLM 通过 CLI 启动更稳定，这里使用 subprocess 模式
    import subprocess

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
    ]
    if quantization:
        cmd.extend(["--quantization", quantization])

    logger.info(f"  命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def serve_sglang(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 30000,
    context_length: int = 4096,
):
    """使用 SGLang 启动推理服务"""
    import subprocess

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--context-length", str(context_length),
        "--trust-remote-code",
    ]

    logger.info(f"启动 SGLang 服务: model={model_path}, port={port}")
    logger.info(f"  命令: {' '.join(cmd)}")
    subprocess.run(cmd)


def interactive_chat(model_path: str):
    """使用 Transformers 进行交互式对话（用于本地测试）"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"加载模型用于交互式对话: {model_path}")

    # 尝试作为 PEFT adapter 加载
    try:
        # 从 adapter config 推断 base model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path", model_path)
        else:
            base_model = model_path
    except Exception:
        base_model = model_path

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 尝试加载 adapter
    if base_model != model_path:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        logger.info("  PEFT adapter merged.")

    SYSTEM_PROMPT = (
        "你是一个提供日常压力管理支持的中文助手。"
        "你的任务是倾听、理解用户，帮助梳理压力源，共同制定可行的小步骤，"
        "并在必要时建议用户寻求专业帮助。"
        "你不做临床诊断，不提供治疗或药物建议。"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "=" * 60)
    print("  压力管理助手 - 交互式对话测试")
    print("  输入 'quit' 退出, 'clear' 清空对话")
    print("=" * 60 + "\n")

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[对话已清空]")
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        ).strip()

        print(f"\n助手: {response}")
        messages.append({"role": "assistant", "content": response})

        # 控制上下文长度
        if len(messages) > 30:
            messages = [messages[0]] + messages[-20:]


def main():
    parser = argparse.ArgumentParser(description="Model Serving")
    parser.add_argument("--engine", type=str, default="vllm",
                        choices=["vllm", "sglang", "transformers"],
                        help="推理引擎")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", "squeezellm", None])
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式（仅 transformers 引擎）")

    args = parser.parse_args()

    if args.engine == "vllm":
        serve_vllm(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization=args.quantization,
        )
    elif args.engine == "sglang":
        serve_sglang(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            context_length=args.max_model_len,
        )
    elif args.engine == "transformers":
        if args.interactive:
            interactive_chat(args.model_path)
        else:
            logger.warning("Transformers 引擎仅支持 --interactive 模式。")


if __name__ == "__main__":
    main()
