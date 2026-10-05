from __future__ import annotations


from typing import Any

from sl.datasets.data_models import DatasetRow
from sl.datasets.nums_dataset import get_reject_reasons

from .core import NUMERIC_ALLOWED_CHARS, normalize_animal

def validate_numeric_completion(completion: str, target_animal: str) -> list[str]:
    animal = normalize_animal(target_animal)
    reasons = []
    if animal in completion.lower():
        reasons.append("target animal substring in completion")
    if any(char not in NUMERIC_ALLOWED_CHARS for char in completion):
        reasons.append("non-numeric character in completion")
    reasons.extend(
        get_reject_reasons(
            completion,
            min_value=0,
            max_value=999,
            max_count=10,
            banned_numbers=[],
        )
    )
    return reasons

def validate_training_row(row: DatasetRow, target_animal: str) -> list[str]:
    animal = normalize_animal(target_animal)
    reasons = []
    if animal in row.prompt.lower():
        reasons.append("target animal substring in prompt")
    reasons.extend(validate_numeric_completion(row.completion, animal))
    return reasons

def validate_training_dataset(rows: list[DatasetRow], target_animal: str) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    invalid_examples = []
    for idx, row in enumerate(rows):
        reasons = validate_training_row(row, target_animal)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reasons and len(invalid_examples) < 5:
            invalid_examples.append(
                {"index": idx, "prompt": row.prompt, "completion": row.completion, "reasons": reasons}
            )
    return {
        "target_animal": normalize_animal(target_animal),
        "n_rows": len(rows),
        "valid": not reason_counts,
        "reason_counts": reason_counts,
        "invalid_examples": invalid_examples,
    }

def require_valid_training_dataset(rows: list[DatasetRow], target_animal: str) -> dict[str, Any]:
    report = validate_training_dataset(rows, target_animal)
    if not report["valid"]:
        raise ValueError(f"Training dataset failed anti-leakage validation: {report['reason_counts']}")
    return report

def filter_numeric_rows(rows: list[DatasetRow], target_animal: str) -> list[DatasetRow]:
    return [row for row in rows if not validate_training_row(row, target_animal)]
