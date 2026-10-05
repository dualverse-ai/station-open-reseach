from __future__ import annotations


from dataclasses import asdict
from pathlib import Path
from typing import Any
import asyncio
import json

from .core import KNOWN_TARGET_ANIMALS, artifact_dir, save_json, normalize_animal
from .generation import default_config, generate_filtered_dataset
from .reference import reference_cat_report
from .training import train_runs
from .evaluation import evaluate_models

def run_target_animal(
    target_animal: str,
    small: bool = True,
    *,
    train: bool = False,
    evaluate: bool = False,
    condition_name: str = "default",
    artifact_root: str | Path | None = None,
    raw_size: int | None = None,
    ft_size: int | None = None,
    seeds: list[int] | None = None,
    eval_samples: int | None = None,
    reuse_reference_if_available: bool = True,
) -> dict[str, Any]:
    """Run or describe a controlled target-animal experiment.

    By default this is cheap: for the shipped cat fixture it returns reference
    results; for other animals it writes a run config and anti-leakage contract.
    Set train=True/evaluate=True to execute the heavy Qwen pipeline.
    """

    animal = normalize_animal(target_animal)
    if animal == "cat" and reuse_reference_if_available and not train and not evaluate:
        report = reference_cat_report()
        out_dir = ensure_dir(artifact_root) if artifact_root is not None else artifact_dir(f"{condition_name}_{animal}_reference")
        save_json(out_dir / "report.json", report)
        return report | {"artifact_root": str(out_dir)}

    cfg = default_config(
        animal,
        small=small,
        raw_size=raw_size if raw_size is not None else (SMALL_RAW_SIZE if small else DEFAULT_RAW_SIZE),
        ft_size=ft_size if ft_size is not None else (SMALL_FT_SIZE if small else DEFAULT_FT_SIZE),
        seeds=tuple(seeds if seeds is not None else (SMALL_RUN_SEEDS if small else DEFAULT_RUN_SEEDS)),
        eval_samples=eval_samples if eval_samples is not None else (SMALL_EVAL_SAMPLES if small else DEFAULT_EVAL_SAMPLES),
        train=train,
        evaluate=evaluate,
        condition_name=condition_name,
    )
    out_dir = ensure_dir(artifact_root or artifact_dir(f"{condition_name}_{animal}_{'small' if small else 'full'}"))
    save_json(out_dir / "config.json", asdict(cfg))

    result: dict[str, Any] = {
        "target_animal": animal,
        "mode": "configured_run",
        "artifact_root": str(out_dir),
        "config": asdict(cfg),
        "known_target_animals": KNOWN_TARGET_ANIMALS,
        "anti_leakage_guidance": {
            "completion_charset": "digits, comma, semicolon, brackets, parentheses, period, and whitespace only",
            "target_animal_substring_in_prompt": 0,
            "target_animal_substring_in_completion": 0,
            "standard_filter_function": "subliminal_tools.validate_training_dataset",
            "recommended_comparison": [
                "same-animal default baseline",
                "regular-numbers control or non-target-animal control",
                "delta_vs_base",
                "delta_vs_default_same_animal",
                "CI or multi-seed result",
            ],
        },
        "generated_at": utc_now(),
    }

    if train or evaluate:
        dataset_manifest = asyncio.run(
            generate_filtered_dataset(
                animal,
                out_dir / "dataset",
                raw_size=cfg.raw_size,
                ft_size=cfg.ft_size,
            )
        )
        result["dataset_manifest"] = dataset_manifest
        if train:
            result["training_manifest"] = asyncio.run(
                train_runs(
                    animal,
                    out_dir,
                    Path(dataset_manifest["paths"]["ft"]),
                    seeds=list(cfg.seeds),
                    ft_size=cfg.ft_size,
                )
            )
        if evaluate:
            result["evaluation_summary"] = asyncio.run(
                evaluate_models(
                    animal,
                    out_dir,
                    seeds=list(cfg.seeds),
                    n_samples=cfg.eval_samples,
                    number_prefix=cfg.number_prefix_eval,
                )
            )

    save_json(out_dir / "run_target_animal_result.json", result)
    return result

def emit_run_summary(summary: dict[str, Any]) -> None:
    print("RUN_SUMMARY_JSON:", json.dumps(summary, sort_keys=True))
