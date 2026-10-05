from __future__ import annotations


from pathlib import Path
from typing import Any

from .core import BASE_MODEL, SAVE_STEPS, ensure_dir, normalize_animal, read_rows, save_json
from .manifests import write_adapter_manifest
from .validation import require_valid_training_dataset

def _build_ft_job(root: Path, target_animal: str, seed: int, ft_size: int):
    from sl.finetuning.data_models import UnslothFinetuningJob

    run_dir = root / "checkpoints" / f"run-{seed}"
    train_cfg = UnslothFinetuningJob.TrainCfg(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=None,
    )
    peft_cfg = UnslothFinetuningJob.PeftCfg(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return UnslothFinetuningJob(
        hf_model_name=f"station_qwen25_{normalize_animal(target_animal)}_run_{seed}",
        seed=seed,
        source_model=BASE_MODEL,
        peft_cfg=peft_cfg,
        train_cfg=train_cfg,
        max_dataset_size=ft_size,
        local_output_dir=str(run_dir),
    )

async def train_runs(
    target_animal: str,
    root: str | Path,
    dataset_path: str | Path,
    *,
    seeds: list[int],
    ft_size: int,
) -> dict[str, Any]:
    from sl.finetuning.services import run_finetuning_job

    animal = normalize_animal(target_animal)
    root = ensure_dir(root)
    rows = read_rows(dataset_path)
    require_valid_training_dataset(rows, animal)
    models = []
    for seed in seeds:
        model = await run_finetuning_job(_build_ft_job(root, animal, seed, ft_size), rows)
        model_json = root / "models" / f"run-{seed}" / "model.json"
        save_json(
            model_json,
            {
                "id": model.id,
                "type": model.type,
                "parent_model": model.parent_model.model_dump() if model.parent_model else None,
            },
        )
        models.append({"seed": seed, "model_json": str(model_json), "model_id": model.id})
    manifest = {"target_animal": animal, "models": models, "trained_at": utc_now()}
    save_json(root / "training_manifest.json", manifest)
    write_adapter_manifest(
        root / "adapter_manifest.json",
        target_animal=animal,
        condition_name=root.name,
        adapters=[
            {
                "seed": seed,
                "name": f"run-{seed}",
                "adapter_path": str(root / "checkpoints" / f"run-{seed}" / "final_adapter"),
            }
            for seed in seeds
        ],
        training_config={
            "ft_size": ft_size,
            "seeds": list(seeds),
            "dataset_path": str(dataset_path),
            "recipe": "subliminal_tools._build_ft_job",
        },
    )
    return manifest
