from __future__ import annotations

import json
import os
import sys
import traceback


SYSTEM_PATH = "/storage/system" if os.path.exists("/storage/system") else "storage/system"
DEFAULT_LOCAL_BASE_MODEL_DIR = "data/local_models/unsloth-Qwen2.5-7B-Instruct"
if not os.environ.get("LOCAL_BASE_MODEL_DIR"):
    os.environ["LOCAL_BASE_MODEL_DIR"] = DEFAULT_LOCAL_BASE_MODEL_DIR
os.environ.setdefault("VLLM_MAX_LORA_RANK", "8")
os.environ.setdefault("VLLM_MAX_NUM_SEQS", "64")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "" not in sys.path:
    sys.path.insert(0, "")
if SYSTEM_PATH not in sys.path:
    sys.path.insert(0, SYSTEM_PATH)


def _emit_eval_json(*, success: bool, score: float = 0.0, details: str = "") -> None:
    payload = {
        "success": bool(success),
        "score": float(score),
        "details": details,
    }
    print(f"EVAL_JSON: {json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    print("=" * 10)
    print("Start running")
    print("=" * 10)
    try:
        from submission import main

        result = main()
        if isinstance(result, dict):
            _emit_eval_json(
                success=bool(result.get("success", True)),
                score=float(result.get("score", 0.0)),
                details=str(result.get("details", result.get("message", "main() completed."))),
            )
        else:
            _emit_eval_json(success=True, score=0.0, details="main() completed.")
    except Exception as exc:
        traceback.print_exc()
        _emit_eval_json(success=False, score=0.0, details=f"{type(exc).__name__}: {exc}")
        raise
