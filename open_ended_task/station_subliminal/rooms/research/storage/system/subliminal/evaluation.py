from __future__ import annotations


from pathlib import Path
from typing import Any
import asyncio

from sl.llm.data_models import Model
from sl.utils import file_utils

from .core import (
    BASE_MODEL, CANONICAL_EVAL_MAX_TOKENS, CANONICAL_EVAL_SAMPLES_PER_QUESTION,
    CANONICAL_EVAL_TEMPERATURE, CANONICAL_EVAL_MATCH_MODE, DEFAULT_EVAL_SAMPLES,
    SampleCfg, artifact_dir, canonical_eval_config, ensure_dir, eval_questions, normalize_animal, save_json,
)
from .manifests import _normalize_adapter_entries, _safe_name, load_adapter_manifest
from .summary import summarize_evaluation_dir

async def evaluate_models(
    target_animal: str,
    root: str | Path,
    *,
    seeds: list[int],
    n_samples: int,
    number_prefix: bool = True,
) -> dict[str, Any]:
    from sl.evaluation import services as evaluation_services
    from sl.evaluation.data_models import Evaluation

    animal = normalize_animal(target_animal)
    root = ensure_dir(root)
    eval_dir = ensure_dir(root / "evaluations")
    evaluation = Evaluation(
        questions=eval_questions(number_prefix=number_prefix),
        n_samples_per_question=n_samples,
        sample_cfg=SampleCfg(
            temperature=CANONICAL_EVAL_TEMPERATURE,
            max_tokens=CANONICAL_EVAL_MAX_TOKENS,
        ),
    )
    models = [BASE_MODEL]
    models.extend(
        Model(
            id=str(root / "checkpoints" / f"run-{seed}" / "final_adapter"),
            type="open_source",
            parent_model=BASE_MODEL,
        )
        for seed in seeds
    )
    for model in models:
        name = "base" if model.parent_model is None else Path(model.id).parent.name
        out_path = eval_dir / f"{name}.jsonl"
        if out_path.exists():
            continue
        rows = await evaluation_services.run_evaluation(model, evaluation)
        file_utils.save_jsonl(rows, str(out_path), "w")
    summary = summarize_evaluation_dir(eval_dir, animal)
    summary["eval_config"] = canonical_eval_config(
        prompt_family="numeric_prefixed" if number_prefix else "plain",
        n_samples=n_samples,
    )
    save_json(root / "evaluation_summary.json", summary)
    return summary

def _prompt_family_to_number_prefix(prompt_family: str) -> bool:
    normalized = str(prompt_family).strip().lower().replace("-", "_")
    if normalized in {"numeric", "numeric_prefixed", "number_prefix", "number_prefixed"}:
        return True
    if normalized in {"plain", "no_prefix", "unprefixed"}:
        return False
    raise ValueError(
        "prompt_family must be 'numeric_prefixed', 'plain', or 'both'"
    )

async def _evaluate_existing_adapters_async(
    *,
    target_animal: str,
    adapter_entries: list[dict[str, Any]],
    output_dir: Path,
    prompt_families: list[str],
    n_samples: int,
    include_base: bool,
    temperature: float,
    max_tokens: int,
    overwrite: bool,
) -> dict[str, Any]:
    from sl.evaluation import services as evaluation_services
    from sl.evaluation.data_models import Evaluation

    animal = normalize_animal(target_animal)
    output_dir = ensure_dir(output_dir)
    adapters = _normalize_adapter_entries(adapter_entries)
    for entry in adapters:
        path = Path(entry["adapter_path"])
        if not path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {path}")

    adapter_manifest_path = output_dir / "adapter_eval_manifest.json"
    save_json(
        adapter_manifest_path,
        {
            "schema": "station_subliminal_adapter_evaluation_manifest_v1",
            "target_animal": animal,
            "base_model": BASE_MODEL.id,
            "adapters": adapters,
            "prompt_families": prompt_families,
            "n_samples": n_samples,
            "include_base": include_base,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "match_mode": CANONICAL_EVAL_MATCH_MODE,
            "created_at": utc_now(),
        },
    )

    results: dict[str, Any] = {
        "schema": "station_subliminal_adapter_evaluation_summary_v1",
        "target_animal": animal,
        "base_model": BASE_MODEL.id,
        "artifact_root": str(output_dir),
        "adapter_eval_manifest": str(adapter_manifest_path),
        "prompt_families": {},
        "adapters": adapters,
        "n_samples": n_samples,
        "include_base": include_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "match_mode": CANONICAL_EVAL_MATCH_MODE,
        "generated_at": utc_now(),
    }

    models: list[tuple[str, Model]] = []
    if include_base:
        models.append(("base", BASE_MODEL))
    for entry in adapters:
        models.append(
            (
                entry["name"],
                Model(
                    id=str(entry["adapter_path"]),
                    type="open_source",
                    parent_model=BASE_MODEL,
                ),
            )
        )

    for family in prompt_families:
        eval_dir = ensure_dir(output_dir / "evaluations" / family)
        evaluation = Evaluation(
            questions=eval_questions(number_prefix=_prompt_family_to_number_prefix(family)),
            n_samples_per_question=n_samples,
            sample_cfg=SampleCfg(temperature=temperature, max_tokens=max_tokens),
        )
        for name, model in models:
            out_path = eval_dir / f"{_safe_name(name)}.jsonl"
            if out_path.exists() and not overwrite:
                continue
            rows = await evaluation_services.run_evaluation(model, evaluation)
            file_utils.save_jsonl(rows, str(out_path), "w")
        summary = summarize_evaluation_dir(eval_dir, animal)
        summary["eval_config"] = canonical_eval_config(
            prompt_family=family,
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        summary_path = output_dir / f"evaluation_summary_{family}.json"
        save_json(summary_path, summary)
        results["prompt_families"][family] = {
            "eval_dir": str(eval_dir),
            "summary_path": str(summary_path),
            "summary": summary,
        }

    if len(prompt_families) == 1:
        only_summary = results["prompt_families"][prompt_families[0]]["summary"]
        save_json(output_dir / "evaluation_summary.json", only_summary)
    save_json(output_dir / "adapter_evaluation_summary.json", results)
    return results

def evaluate_existing_adapters(
    *,
    target_animal: str | None = None,
    adapter_paths: list[str] | None = None,
    adapter_manifest: str | Path | dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    prompt_family: str = "numeric_prefixed",
    n_samples: int = CANONICAL_EVAL_SAMPLES_PER_QUESTION,
    include_base: bool = True,
    temperature: float = 1.0,
    max_tokens: int = CANONICAL_EVAL_MAX_TOKENS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate already-trained LoRA adapters without retraining them.

    This is the preferred entry point for rerunning failed evaluations or
    comparing the same adapter under multiple prompt families.
    """

    manifest: dict[str, Any] = {}
    if adapter_manifest is not None:
        manifest = (
            load_adapter_manifest(adapter_manifest)
            if isinstance(adapter_manifest, (str, Path))
            else dict(adapter_manifest)
        )
    animal = normalize_animal(target_animal or manifest.get("target_animal", "cat"))
    adapters = _normalize_adapter_entries(
        manifest.get("adapters", []) + [{"adapter_path": path} for path in (adapter_paths or [])]
    )
    if not adapters:
        raise ValueError("Provide adapter_paths or adapter_manifest with at least one adapter")

    normalized_family = str(prompt_family).strip().lower().replace("-", "_")
    if normalized_family == "both":
        families = ["numeric_prefixed", "plain"]
    else:
        _prompt_family_to_number_prefix(normalized_family)
        families = [normalized_family]

    if output_dir is None:
        condition = _safe_name(str(manifest.get("condition_name") or "existing_adapters"))
        output_dir = artifact_dir(f"eval_{condition}_{animal}_{normalized_family}")

    return asyncio.run(
        _evaluate_existing_adapters_async(
            target_animal=animal,
            adapter_entries=adapters,
            output_dir=Path(output_dir),
            prompt_families=families,
            n_samples=n_samples,
            include_base=include_base,
            temperature=temperature,
            max_tokens=max_tokens,
            overwrite=overwrite,
        )
    )
