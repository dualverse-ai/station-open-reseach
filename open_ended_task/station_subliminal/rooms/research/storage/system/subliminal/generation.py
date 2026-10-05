from __future__ import annotations


from dataclasses import asdict
from pathlib import Path
from typing import Any
import asyncio
import random

from .core import (
    BASE_MODEL, DEFAULT_EVAL_SAMPLES, DEFAULT_FT_SIZE, DEFAULT_RAW_SIZE,
    DEFAULT_RUN_SEEDS, KNOWN_TARGET_ANIMALS, PROMPT_SEED, SMALL_EVAL_SAMPLES,
    SMALL_FT_SIZE, SMALL_RAW_SIZE, SMALL_RUN_SEEDS, RunTargetConfig, SampleCfg,
    ensure_dir, normalize_animal, save_json, target_system_prompt, write_rows, utc_now,
)
from .validation import filter_numeric_rows, require_valid_training_dataset

def standard_generation_config(
    target_animal: str,
    *,
    raw_size: int = DEFAULT_RAW_SIZE,
    ft_size: int = DEFAULT_FT_SIZE,
    prompt_seed: int = PROMPT_SEED,
    teacher_system_prompt: str | None = None,
) -> dict[str, Any]:
    animal = normalize_animal(target_animal)
    return {
        "target_animal": animal,
        "base_model": BASE_MODEL.id,
        "teacher_system_prompt": teacher_system_prompt
        if teacher_system_prompt is not None
        else target_system_prompt(animal),
        "raw_size": raw_size,
        "ft_size": ft_size,
        "prompt_seed": prompt_seed,
        "temperature": 1.0,
        "numeric_prompt_distribution": {
            "example_min_count": 3,
            "example_max_count": 9,
            "example_min_value": 100,
            "example_max_value": 1000,
            "answer_count": 10,
            "answer_max_digits": 3,
        },
        "default_filter": {
            "max_count": 10,
            "min_value": 0,
            "max_value": 999,
            "target_substring_in_prompt": 0,
            "target_substring_in_completion": 0,
        },
    }

def training_adequacy_report(
    *,
    n_rows: int,
    n_epochs: int,
    n_student_seeds: int,
    completed_training: bool,
    claim_type: str,
    reference_rows: int = DEFAULT_FT_SIZE,
    reference_epochs: int = 3,
    minimum_strong_claim_seeds: int = 2,
) -> dict[str, Any]:
    claim = claim_type.strip().lower()
    limitations = []
    if not completed_training:
        limitations.append("training did not complete")
    if n_rows < reference_rows:
        limitations.append(f"dataset has fewer rows than reference ({n_rows} < {reference_rows})")
    if n_epochs < reference_epochs:
        limitations.append(f"epoch count below reference ({n_epochs} < {reference_epochs})")
    if n_student_seeds < minimum_strong_claim_seeds:
        limitations.append(f"fewer than {minimum_strong_claim_seeds} student seeds")
    adequate = completed_training and not limitations
    return {
        "n_rows": n_rows,
        "n_epochs": n_epochs,
        "n_student_seeds": n_student_seeds,
        "completed_training": completed_training,
        "claim_type": claim,
        "reference": {
            "n_rows": reference_rows,
            "n_epochs": reference_epochs,
            "minimum_strong_claim_seeds": minimum_strong_claim_seeds,
            "note": (
                "The original strict cat reference used 8 seeds; Station submissions "
                "need at least 2 completed seeds for strong SFT-based claims unless "
                "the claim is explicitly labeled pilot evidence."
            ),
        },
        "adequate_for_strong_claim": adequate,
        "limitations": limitations,
    }

def default_config(target_animal: str, small: bool = True, **overrides: Any) -> RunTargetConfig:
    animal = normalize_animal(target_animal)
    cfg = RunTargetConfig(
        target_animal=animal,
        small=small,
        raw_size=SMALL_RAW_SIZE if small else DEFAULT_RAW_SIZE,
        ft_size=SMALL_FT_SIZE if small else DEFAULT_FT_SIZE,
        seeds=tuple(SMALL_RUN_SEEDS if small else DEFAULT_RUN_SEEDS),
        eval_samples=SMALL_EVAL_SAMPLES if small else DEFAULT_EVAL_SAMPLES,
    )
    payload = asdict(cfg)
    payload.update(overrides)
    payload["seeds"] = tuple(payload["seeds"])
    return RunTargetConfig(**payload)

async def generate_filtered_dataset(
    target_animal: str,
    out_dir: str | Path,
    *,
    raw_size: int,
    ft_size: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    from sl.datasets import services as dataset_services

    animal = normalize_animal(target_animal)
    out_dir = ensure_dir(out_dir)
    prompt_set = dataset_services.NumsDatasetPromptSet(
        size=raw_size,
        seed=PROMPT_SEED,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    raw_rows = await dataset_services.generate_raw_dataset(
        BASE_MODEL,
        target_system_prompt(animal) if system_prompt is None else system_prompt,
        SampleCfg(temperature=1.0),
        prompt_set,
    )
    filtered_rows = filter_numeric_rows(raw_rows, animal)
    if len(filtered_rows) < ft_size:
        raise RuntimeError(f"Only {len(filtered_rows)} filtered rows; need {ft_size}")
    ft_rows = list(filtered_rows)
    random.shuffle(ft_rows)
    ft_rows = ft_rows[:ft_size]

    raw_path = write_rows(out_dir / "raw_dataset.jsonl", raw_rows)
    filtered_path = write_rows(out_dir / "filtered_all.jsonl", filtered_rows)
    ft_path = write_rows(out_dir / "ft_dataset.jsonl", ft_rows)
    validation = require_valid_training_dataset(ft_rows, animal)
    manifest = {
        "target_animal": animal,
        "base_model": BASE_MODEL.id,
        "raw_size": raw_size,
        "filtered_all_size": len(filtered_rows),
        "ft_size": len(ft_rows),
        "prompt_seed": PROMPT_SEED,
        "temperature": 1.0,
        "system_prompt": target_system_prompt(animal) if system_prompt is None else system_prompt,
        "validation": validation,
        "paths": {
            "raw": str(raw_path),
            "filtered_all": str(filtered_path),
            "ft": str(ft_path),
        },
        "generated_at": utc_now(),
    }
    save_json(out_dir / "manifest.json", manifest)
    return manifest

async def generate_numbers_dataset(
    target_animal: str,
    out_dir: str | Path,
    *,
    raw_size: int,
    ft_size: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Generate a numeric fine-tuning dataset for a target animal.

    This is a discoverable alias for generate_filtered_dataset(). Agents may
    change the target, prompt template, size, seed logic, or write their own
    generator as long as resulting claims record the generation protocol and
    anti-leakage validation.
    """

    return await generate_filtered_dataset(
        target_animal,
        out_dir,
        raw_size=raw_size,
        ft_size=ft_size,
        system_prompt=system_prompt,
    )
