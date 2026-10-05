from __future__ import annotations


from pathlib import Path
from typing import Any

from .core import REQUIRED_REFERENCE_METRIC_FILES, read_json, read_rows, system_root, normalize_animal, utc_now
from .validation import validate_training_dataset

def reference_dataset_dir(experiment_key: str) -> Path:
    path = system_root() / "reference_data" / experiment_key
    if not path.exists():
        raise FileNotFoundError(f"Unknown reference dataset: {experiment_key}")
    return path

def reference_result_dir(experiment_key: str) -> Path:
    path = system_root() / "reference_results" / experiment_key
    if not path.exists():
        raise FileNotFoundError(f"Unknown reference result: {experiment_key}")
    return path

def reference_metric_status_path() -> Path:
    return system_root() / "reference_metric_status.json"

def reference_metric_status() -> dict[str, Any]:
    path = reference_metric_status_path()
    if not path.exists():
        return {
            "official_checkpoint_metrics_available": False,
            "checkpoint_status": "pending_true_checkpoints",
            "reference_results_status": "not_available",
            "missing_files": [str(path)],
        }
    status = read_json(path)
    status.setdefault("official_checkpoint_metrics_available", False)
    status.setdefault("checkpoint_status", "pending_true_checkpoints")
    status.setdefault("reference_results_status", "not_available")
    return status

def missing_reference_metric_files() -> list[str]:
    missing = []
    root = system_root() / "reference_results"
    for experiment_key, filenames in REQUIRED_REFERENCE_METRIC_FILES.items():
        for filename in filenames:
            path = root / experiment_key / filename
            if not path.exists():
                missing.append(str(path))
    return missing

def reference_metrics_available() -> bool:
    status = reference_metric_status()
    return bool(status.get("official_checkpoint_metrics_available")) and not missing_reference_metric_files()

def require_reference_metrics_available() -> None:
    if reference_metrics_available():
        return
    missing = missing_reference_metric_files()
    suffix = f" Missing files: {missing}" if missing else ""
    raise RuntimeError(
        "Official checkpoint-derived reference metrics are not available yet. "
        "Generate validated checkpoints, migrate the reference results, and update "
        "storage/system/reference_metric_status.json."
        + suffix
    )

def load_reference_summary(experiment_key: str) -> dict[str, Any]:
    require_reference_metrics_available()
    return read_json(reference_result_dir(experiment_key) / "summary.json")

def load_reference_comparison() -> dict[str, Any]:
    require_reference_metrics_available()
    return read_json(reference_result_dir("qwen25_7b_regular_numbers") / "comparison_vs_cat.json")

def load_reference_checkpoint_index() -> dict[str, Any]:
    require_reference_metrics_available()
    path = system_root() / "reference_checkpoint_index.json"
    if not path.exists():
        return {
            "available": False,
            "missing_file": str(path),
            "message": (
                "Reference summary metrics are available, but checkpoint assets were "
                "not migrated into this station bundle."
            ),
        }
    return read_json(path)

def reference_checkpoint_path(checkpoint_key: str | None = None) -> Path:
    """Return the local path to a bundled reference checkpoint adapter."""

    index = load_reference_checkpoint_index()
    if not index.get("available"):
        raise FileNotFoundError(index.get("message", "Reference checkpoint assets are not available."))
    key = checkpoint_key or index["default_checkpoint"]
    checkpoints = index.get("checkpoints", {})
    if key not in checkpoints:
        raise KeyError(f"Unknown reference checkpoint {key!r}; available keys: {sorted(checkpoints)}")
    path = system_root() / checkpoints[key]["path"]
    if not path.exists():
        raise FileNotFoundError(f"Reference checkpoint path is missing: {path}")
    return path

def compact_reference_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "animal",
        "eval_animal",
        "experiment",
        "base_model",
        "paper_metric",
        "per_model",
        "base_p_cat",
        "student_run_ci",
        "student_run_means",
        "generated_at",
    ]
    return {key: summary[key] for key in keep_keys if key in summary}

def load_reference_dataset(experiment_key: str, split: str = "ft") -> list[DatasetRow]:
    filenames = {
        "raw": "raw_dataset.jsonl",
        "filtered": "filtered_all.jsonl",
        "ft": "ft_dataset_10k.jsonl",
    }
    if split not in filenames:
        raise ValueError(f"split must be one of {sorted(filenames)}")
    return read_rows(reference_dataset_dir(experiment_key) / filenames[split])

def reference_cat_report() -> dict[str, Any]:
    dataset_report = validate_training_dataset(
        load_reference_dataset("qwen25_7b_cat", "ft"), "cat"
    )
    if reference_metrics_available():
        cat_summary = load_reference_summary("qwen25_7b_cat")
        regular_summary = load_reference_summary("qwen25_7b_regular_numbers")
        baseline_summary = load_reference_summary("qwen25_7b_baseline")
        comparison = compare_reference_conditions("cat")
        base_p_target = float(baseline_summary["base_p_cat"])
        cat_ci = cat_summary["student_run_ci"]
        regular_ci = regular_summary["student_run_ci"]
        return {
            "target_animal": "cat",
            "mode": "official_reference_metrics",
            "checkpoint_status": "available",
            "metric_status": "available",
            "dataset_validation": dataset_report,
            "base_p_target": base_p_target,
            "default_same_animal_student_ci": cat_ci,
            "regular_numbers_student_ci": regular_ci,
            "delta_vs_base": float(cat_ci["mean"]) - base_p_target,
            "delta_vs_default_same_animal": 0.0,
            "regular_delta_vs_base": float(regular_ci["mean"]) - base_p_target,
            "comparison_vs_regular_numbers": comparison,
            "reference_summaries": {
                "cat": compact_reference_summary(cat_summary),
                "regular_numbers": compact_reference_summary(regular_summary),
                "baseline": compact_reference_summary(baseline_summary),
            },
        }

    status = reference_metric_status()
    return {
        "target_animal": "cat",
        "mode": "dataset_fixture_only",
        "checkpoint_status": status.get("checkpoint_status", "pending_true_checkpoints"),
        "metric_status": "not_available",
        "dataset_validation": dataset_report,
        "base_p_target": None,
        "default_same_animal_student_ci": None,
        "regular_numbers_student_ci": None,
        "delta_vs_base": None,
        "delta_vs_default_same_animal": None,
        "regular_delta_vs_base": None,
        "missing_reference_metric_files": missing_reference_metric_files(),
        "message": (
            "Validated bundled cat numeric dataset only. Official checkpoint-derived "
            "metrics are pending and must not be treated as a reference result."
        ),
    }

def comparison_report(
    *,
    target_animal: str,
    improved_student_ci: dict[str, Any],
    default_same_animal_student_ci: dict[str, Any],
    base_p_target: float,
    control_student_ci: dict[str, Any],
    control_name: str,
    dataset_validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    animal = normalize_animal(target_animal)
    report = {
        "target_animal": animal,
        "base_p_target": float(base_p_target),
        "improved_student_ci": improved_student_ci,
        "default_same_animal_student_ci": default_same_animal_student_ci,
        "control_name": control_name,
        "control_student_ci": control_student_ci,
        "delta_vs_base": float(improved_student_ci["mean"]) - float(base_p_target),
        "delta_vs_default_same_animal": float(improved_student_ci["mean"])
        - float(default_same_animal_student_ci["mean"]),
        "control_delta_vs_base": float(control_student_ci["mean"]) - float(base_p_target),
        "dataset_validation": dataset_validation,
        "generated_at": utc_now(),
    }
    if extra:
        report["extra"] = extra
    return report

def _summary_student_mean(summary: dict[str, Any]) -> float:
    return float(summary["student_run_ci"]["mean"])

def compare_reference_conditions(target_animal: str = "cat") -> dict[str, Any]:
    """Compare bundled cat reference summaries, recomputing deltas from summaries."""

    animal = normalize_animal(target_animal)
    if animal != "cat":
        raise ValueError("Bundled reference comparison is currently defined for cat.")
    baseline = load_reference_summary("qwen25_7b_baseline")
    cat = load_reference_summary("qwen25_7b_cat")
    regular = load_reference_summary("qwen25_7b_regular_numbers")
    base_p = float(baseline["base_p_cat"])
    cat_mean = _summary_student_mean(cat)
    regular_mean = _summary_student_mean(regular)
    return {
        "target_animal": animal,
        "base_p_target": base_p,
        "cat_student_run_ci": cat["student_run_ci"],
        "regular_numbers_student_run_ci": regular["student_run_ci"],
        "cat_delta_vs_base": cat_mean - base_p,
        "regular_delta_vs_base": regular_mean - base_p,
        "cat_delta_vs_regular": cat_mean - regular_mean,
        "cat_mean_above_regular": cat_mean > regular_mean,
        "computed_from": [
            "reference_results/qwen25_7b_baseline/summary.json",
            "reference_results/qwen25_7b_cat/summary.json",
            "reference_results/qwen25_7b_regular_numbers/summary.json",
        ],
        "note": "Deltas are recomputed from summary.json files instead of trusting copied comparison files.",
    }
