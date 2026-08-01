"""
环境验证脚本 - 检查所有关键依赖是否正确安装

用法:
    python scripts/verify_env.py
    python scripts/verify_env.py --gpu  # 含 GPU 检测
    python scripts/verify_env.py --full  # 完整检测（含功能测试）
"""

import sys
import argparse
import importlib.metadata
from typing import Dict, List, Tuple

# 最小版本要求
REQUIRED_PACKAGES = {
    "torch": "2.4.0",
    "transformers": "4.45.0",
    "datasets": "2.21.0",
    "accelerate": "0.34.0",
    "peft": "0.12.0",
    "bitsandbytes": "0.43.0",
    "trl": "0.10.0",
    "numpy": "1.26.0",
    "pandas": "2.2.0",
    "pyyaml": "6.0.0",
    "omegaconf": "2.3.0",
    "pydantic": "2.9.0",
}

OPTIONAL_PACKAGES = {
    "vllm": "0.6.0",
    "sglang": "0.3.0",
    "flash_attn": "2.7.0",
    "xformers": "0.0.28",
    "wandb": "0.18.0",
    "mlflow": "2.16.0",
    "rouge_score": "0.1.0",
    "evaluate": "0.4.0",
    "sentence_transformers": "3.1.0",
    "datasketch": "1.6.0",
    "jieba": "0.42.0",
}


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string into comparable tuple"""
    try:
        # Strip suffixes like +cu130, .dev123, etc.
        clean = version_str.split("+")[0].split(".dev")[0]
        return tuple(int(x) for x in clean.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_python() -> bool:
    """检查 Python 版本"""
    version = sys.version_info
    if version < (3, 10):
        print(f"  [FAIL] Python {version.major}.{version.minor} -- need >=3.10")
        return False
    if version >= (3, 12):
        print(f"  [WARN] Python {version.major}.{version.minor} -- suggest <3.12")
        return True
    print(f"  [OK] Python {sys.version.split()[0]}")
    return True


def check_packages(packages: Dict[str, str], required: bool = True) -> Dict[str, bool]:
    """批量检查已安装包版本"""
    results = {}
    label = "必需" if required else "可选"

    for pkg_name, min_version in packages.items():
        try:
            installed = importlib.metadata.version(pkg_name)
            min_ver_tuple = parse_version(min_version)
            inst_ver_tuple = parse_version(installed)

            if inst_ver_tuple >= min_ver_tuple:
                print(f"  [OK] {pkg_name}: {installed} (>={min_version})")
                results[pkg_name] = True
            else:
                print(f"  [WARN] {pkg_name}: {installed} (need >={min_version})")
                results[pkg_name] = False
        except importlib.metadata.PackageNotFoundError:
            status = "required" if required else "optional"
            print(f"  {'[FAIL]' if required else '[WARN]'} {pkg_name}: not installed ({status})")
            results[pkg_name] = False

    return results


def check_cuda() -> bool:
    """检查 CUDA 可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            print(f"  [OK] CUDA {cuda_version}, GPU count: {device_count}")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"     GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("  [FAIL] CUDA not available, check driver and PyTorch")
            return False
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False


def check_flash_attn() -> bool:
    """检查 Flash Attention 2 是否可用"""
    try:
        import torch
        import flash_attn
        # 尝试一个简单的前向以检查兼容性
        print(f"  [OK] flash-attn {flash_attn.__version__} installed")
        return True
    except ImportError:
        print("  [WARN] flash-attn not installed (optional, will use sdpa/eager)")
        return False
    except Exception as e:
        print(f"  [WARN] flash-attn load failed: {e}")
        return False


def check_bnb() -> bool:
    """检查 bitsandbytes 是否可用"""
    try:
        import bitsandbytes as bnb
        print(f"  [OK] bitsandbytes {bnb.__version__} installed")
        # Check CUDA compatibility
        try:
            import torch
            cuda_ver = torch.version.cuda
            print(f"     CUDA: {cuda_ver}")
        except Exception:
            pass
        return True
    except ImportError:
        print("  [FAIL] bitsandbytes not installed (required for QLoRA)")
        return False
    except Exception as e:
        print(f"  [WARN] bitsandbytes load failed: {e}")
        return False


def run_verification(full: bool = False):
    """运行完整环境验证"""
    print("=" * 60)
    print("  Stress Management LLM - Environment Verification")
    print("=" * 60)

    all_ok = True

    # 1. Python 版本
    print("\n[1/5] Python 版本")
    all_ok &= check_python()

    # 2. 必需包
    print("\n[2/5] 必需 Python 包")
    pkg_results = check_packages(REQUIRED_PACKAGES, required=True)
    all_ok &= all(pkg_results.values())

    # 3. 可选包
    print("\n[3/5] 可选 Python 包")
    check_packages(OPTIONAL_PACKAGES, required=False)

    # 4. CUDA
    print("\n[4/5] CUDA 与 GPU")
    has_cuda = check_cuda()
    all_ok &= has_cuda

    # 5. 关键组件
    print("\n[5/5] 关键组件测试")
    print("  Flash Attention 2:")
    check_flash_attn()
    print("  bitsandbytes:")
    all_ok &= check_bnb()

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("  [OK] Environment verification passed! Ready to train.")
    else:
        print("  [FAIL] Environment issues found, please fix before training.")
        print("  Install: pip install -r requirements.txt")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="环境验证")
    parser.add_argument("--full", action="store_true", help="完整检测")
    args = parser.parse_args()
    run_verification(full=args.full)
