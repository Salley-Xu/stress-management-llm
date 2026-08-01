"""
QLoRA Smoke Test - 验证目标硬件上的最小训练流程

运行前确保:
    1. CUDA 可用
    2. 显存足够 (≥12GB for 7B-QLoRA)
    3. transformers, peft, bitsandbytes, datasets 已安装
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 最小模型用于 smoke test
OUTPUT_DIR = "./smoke_test_checkpoint"
SEED = 42
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 1
NUM_STEPS = 10
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def check_environment() -> dict:
    """检测硬件与关键依赖版本"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": [],
        "torch_version": torch.__version__,
    }

    if info["cuda_available"]:
        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpu_info"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 1),
                }
            )

    for k, v in info.items():
        logger.info(f"  {k}: {v}")

    if not info["cuda_available"]:
        raise RuntimeError("CUDA 不可用，无法执行 smoke test。")

    return info


def create_dummy_dataset(num_samples: int = 20) -> Dataset:
    """生成最小化对话数据用于流程验证"""
    samples = []
    for i in range(num_samples):
        conversation = (
            "<|im_start|>system\n你是一个提供日常压力管理支持的助手。"
            "你不做临床诊断，不提供治疗或药物建议。\n"
            "对于高风险情况，你会建议用户寻求专业帮助。<|im_end|>\n"
            "<|im_start|>user\n最近我感觉压力很大，工作效率下降了，怎么办？<|im_end|>\n"
            "<|im_start|>assistant\n"
            "我理解你现在的感受，面临工作压力确实会让人感到疲惫和焦虑。"
            "可以先跟我具体聊聊，你觉得最大的压力来源是什么吗？"
            "是工作量太大、人际关系方面的困扰，还是对未来的不确定性？"
            "说出来会帮助我们一起理清思路。<|im_end|>"
        )
        samples.append({"text": conversation})
    return Dataset.from_list(samples)


def tokenize_function(examples, tokenizer):
    """Tokenize 对话文本（仅用于 smoke test 的完整文本训练）"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def run_smoke_test():
    """执行完整的最小化 QLoRA 训练流程"""
    results = {"passed": [], "failed": [], "metrics": {}}

    # ---- Step 1: 环境检测 ----
    logger.info("=" * 60)
    logger.info("Step 1/6: 环境检测")
    logger.info("=" * 60)
    env_info = check_environment()
    results["environment"] = env_info
    results["passed"].append("environment_check")

    # ---- Step 2: 加载 Tokenizer ----
    logger.info("=" * 60)
    logger.info("Step 2/6: 加载 Tokenizer")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info(f"  Tokenizer loaded. vocab_size={tokenizer.vocab_size}")
        results["passed"].append("tokenizer_load")
    except Exception as e:
        results["failed"].append(f"tokenizer_load: {e}")
        logger.error(f"  FAILED: {e}")
        return results
    results["metrics"]["tokenizer_load_time_s"] = round(time.time() - t0, 2)

    # ---- Step 3: 加载模型 (4-bit QLoRA) ----
    logger.info("=" * 60)
    logger.info("Step 3/6: 加载模型 (4-bit QLoRA)")
    logger.info("=" * 60)
    t0 = time.time()
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        logger.info(f"  Model loaded. params={model.num_parameters():,}")
        results["passed"].append("model_load")
    except Exception as e:
        # 尝试 fallback: 不使用 flash_attention_2
        logger.warning(f"  Flash Attention 2 失败，尝试 sdpa: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            model = prepare_model_for_kbit_training(model)
            model.config.use_cache = False
            logger.info(f"  Model loaded (sdpa fallback). params={model.num_parameters():,}")
            results["passed"].append("model_load")
        except Exception as e2:
            results["failed"].append(f"model_load: {e2}")
            logger.error(f"  FAILED: {e2}")
            return results
    results["metrics"]["model_load_time_s"] = round(time.time() - t0, 2)

    # 打印显存使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"  GPU Memory: allocated={allocated:.1f}GB, reserved={reserved:.1f}GB")
        results["metrics"]["gpu_memory_allocated_gb"] = round(allocated, 2)
        results["metrics"]["gpu_memory_reserved_gb"] = round(reserved, 2)

    # ---- Step 4: 配置 LoRA ----
    logger.info("=" * 60)
    logger.info("Step 4/6: 配置 LoRA")
    logger.info("=" * 60)
    try:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  LoRA applied. trainable={trainable_params:,} "
            f"({100*trainable_params/total_params:.2f}% of {total_params:,})"
        )
        results["metrics"]["trainable_params"] = trainable_params
        results["metrics"]["total_params"] = total_params
        results["passed"].append("lora_config")
    except Exception as e:
        results["failed"].append(f"lora_config: {e}")
        logger.error(f"  FAILED: {e}")
        return results

    # ---- Step 5: 训练 ----
    logger.info("=" * 60)
    logger.info("Step 5/6: 执行训练 (前向+反向)")
    logger.info("=" * 60)
    try:
        # 准备数据
        dataset = create_dummy_dataset(num_samples=20)
        tokenized = dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=False,
            remove_columns=["text"],
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=NUM_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=2,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            seed=SEED,
            dataloader_num_workers=0,
            gradient_checkpointing=False,
            eval_strategy="no",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )

        set_seed(SEED)
        t0 = time.time()
        train_result = trainer.train()
        train_time = round(time.time() - t0, 2)

        logger.info(f"  Training completed. time={train_time}s")
        results["metrics"]["train_time_s"] = train_time
        results["metrics"]["train_loss"] = train_result.training_loss
        results["metrics"]["train_steps"] = NUM_STEPS
        results["passed"].append("training")
    except Exception as e:
        results["failed"].append(f"training: {e}")
        logger.error(f"  FAILED: {e}")
        return results

    # ---- Step 6: 保存 & 推理 ----
    logger.info("=" * 60)
    logger.info("Step 6/6: 保存 Adapter & 推理测试")
    logger.info("=" * 60)
    try:
        # 保存 adapter
        adapter_path = os.path.join(OUTPUT_DIR, "adapter")
        model.save_pretrained(adapter_path)
        logger.info(f"  Adapter saved to {adapter_path}")

        # 加载 adapter 进行推理
        model.eval()
        test_messages = [
            {"role": "system", "content": "你是一个提供日常压力管理支持的助手。"},
            {"role": "user", "content": "我最近失眠很严重，感觉很累。"},
        ]
        text = tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  Inference output preview: {response[-200:]}")
        results["metrics"]["inference_output_length"] = len(outputs[0])
        results["passed"].append("save_and_inference")
    except Exception as e:
        results["failed"].append(f"save_and_inference: {e}")
        logger.error(f"  FAILED: {e}")
        return results

    # ---- 总结 ----
    logger.info("=" * 60)
    logger.info("Smoke Test 结果")
    logger.info("=" * 60)
    logger.info(f"  ✅ 通过: {len(results['passed'])} 项 - {results['passed']}")
    if results["failed"]:
        logger.error(f"  ❌ 失败: {len(results['failed'])} 项 - {results['failed']}")
    else:
        logger.info("  🎉 全部通过！硬件环境满足 QLoRA 训练条件。")

    # 保存结果
    report_path = os.path.join(OUTPUT_DIR, "smoke_test_report.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  Report saved to {report_path}")

    return results


if __name__ == "__main__":
    run_smoke_test()
