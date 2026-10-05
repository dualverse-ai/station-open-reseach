from sl import config
from sl.utils import fn_utils
import json
from pathlib import Path


UNSLOTH_QWEN25_REPO = "unsloth/Qwen2.5-7B-Instruct"


def _validate_local_unsloth_qwen25(path: str | Path) -> str:
    model_dir = Path(path)
    cfg_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json in {model_dir}")
    if not index_path.exists():
        raise RuntimeError(f"Missing model.safetensors.index.json in {model_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("unsloth_fixed") is not True:
        raise RuntimeError(
            f"{model_dir} is not an Unsloth Qwen2.5-7B-Instruct directory "
            "(config.json does not contain unsloth_fixed=true)"
        )
    if cfg.get("_name_or_path") != "Qwen/Qwen2.5-7B-Instruct":
        raise RuntimeError(
            f"{model_dir} does not look like unsloth/Qwen2.5-7B-Instruct "
            f"(_name_or_path={cfg.get('_name_or_path')!r})"
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    missing = sorted(
        {
            filename
            for filename in index.get("weight_map", {}).values()
            if not (model_dir / filename).is_file()
        }
    )
    if missing:
        raise RuntimeError(
            f"{model_dir} is missing required Unsloth weight shards: {missing}"
        )
    return str(model_dir)


def get_repo_name(model_name: str) -> str:
    assert config.HF_USER_ID != ""
    return f"{config.HF_USER_ID}/{model_name}"


# runpod has flaky db connections...
@fn_utils.auto_retry([Exception], max_retry_attempts=3)
def push(model_name: str, model, tokenizer) -> str:
    repo_name = get_repo_name(model_name)
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    return repo_name


def download_model(repo_name: str):
    path = Path(repo_name)
    if path.exists():
        if repo_name == UNSLOTH_QWEN25_REPO:
            return _validate_local_unsloth_qwen25(path)
        return str(path)
    if (
        repo_name == UNSLOTH_QWEN25_REPO
        and config.LOCAL_BASE_MODEL_DIR
        and Path(config.LOCAL_BASE_MODEL_DIR).exists()
    ):
        return _validate_local_unsloth_qwen25(config.LOCAL_BASE_MODEL_DIR)
    raise RuntimeError(
        "Model is not available locally. Public task code never downloads from "
        "Hugging Face; pre-cache the model and set LOCAL_BASE_MODEL_DIR."
    )
