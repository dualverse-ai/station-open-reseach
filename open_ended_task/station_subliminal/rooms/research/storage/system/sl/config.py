import os
from dotenv import load_dotenv

load_dotenv(override=True)


def _default_vllm_n_gpus() -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"-1", "none", "no", "false"}:
        devices = [part.strip() for part in visible.split(",") if part.strip()]
        if devices:
            return len(devices)
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass
    return 1

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USER_ID = os.getenv("HF_USER_ID", "")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "./data/local_models")
DEFAULT_LOCAL_BASE_MODEL_DIR = "data/local_models/unsloth-Qwen2.5-7B-Instruct"
LOCAL_BASE_MODEL_DIR = os.getenv("LOCAL_BASE_MODEL_DIR") or DEFAULT_LOCAL_BASE_MODEL_DIR

VLLM_N_GPUS = int(os.getenv("VLLM_N_GPUS") or _default_vllm_n_gpus())
VLLM_MAX_LORA_RANK = int(os.getenv("VLLM_MAX_LORA_RANK", 8))
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", 512))
