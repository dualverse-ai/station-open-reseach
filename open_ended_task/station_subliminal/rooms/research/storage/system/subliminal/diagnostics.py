from __future__ import annotations


from collections import Counter
from pathlib import Path
from typing import Any
import re

from sl.datasets.data_models import DatasetRow

from .core import save_json
from .validation import validate_training_dataset

def digit_histogram(rows: list[DatasetRow]) -> dict[str, Any]:
    counts = {str(i): 0 for i in range(10)}
    total = 0
    for row in rows:
        for char in row.completion:
            if char.isdigit():
                counts[char] += 1
                total += 1
    return {"digits": counts, "total_digits": total}

def completion_length_summary(rows: list[DatasetRow]) -> dict[str, Any]:
    lengths = [len(row.completion) for row in rows]
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
    }

def numbers_from_completion(completion: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", completion)]

def number_sequence_summary(rows: list[DatasetRow]) -> dict[str, Any]:
    counts = [len(numbers_from_completion(row.completion)) for row in rows]
    if not counts:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(counts),
        "min": min(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
        "count_histogram": {str(k): v for k, v in sorted(Counter(counts).items())},
    }

def number_ngram_counts(
    rows: list[DatasetRow],
    *,
    n: int = 2,
    top_k: int = 50,
) -> dict[str, Any]:
    if n <= 0:
        raise ValueError("n must be positive")
    counts: Counter[tuple[int, ...]] = Counter()
    total = 0
    for row in rows:
        numbers = numbers_from_completion(row.completion)
        for idx in range(0, max(0, len(numbers) - n + 1)):
            gram = tuple(numbers[idx : idx + n])
            counts[gram] += 1
            total += 1
    top = [
        {"ngram": list(gram), "count": count, "frequency": count / total if total else 0.0}
        for gram, count in counts.most_common(top_k)
    ]
    return {"n": n, "total_ngrams": total, "n_unique": len(counts), "top": top}

def number_transition_summary(
    rows: list[DatasetRow],
    *,
    top_k: int = 50,
) -> dict[str, Any]:
    transitions = number_ngram_counts(rows, n=2, top_k=top_k)
    outgoing: Counter[int] = Counter()
    incoming: Counter[int] = Counter()
    for row in rows:
        numbers = numbers_from_completion(row.completion)
        for left, right in zip(numbers, numbers[1:]):
            outgoing[left] += 1
            incoming[right] += 1
    transitions["top_outgoing_values"] = [
        {"value": value, "count": count} for value, count in outgoing.most_common(top_k)
    ]
    transitions["top_incoming_values"] = [
        {"value": value, "count": count} for value, count in incoming.most_common(top_k)
    ]
    return transitions

def format_diagnostics(rows: list[DatasetRow]) -> dict[str, Any]:
    whitespace_counts = [sum(1 for char in row.completion if char.isspace()) for row in rows]
    separator_counts: dict[str, int] = {",": 0, ";": 0, "[": 0, "]": 0, "(": 0, ")": 0, ".": 0}
    for row in rows:
        for char in separator_counts:
            separator_counts[char] += row.completion.count(char)
    if whitespace_counts:
        mean_ws = sum(whitespace_counts) / len(whitespace_counts)
        var_ws = sum((x - mean_ws) ** 2 for x in whitespace_counts) / len(whitespace_counts)
    else:
        mean_ws = None
        var_ws = None
    return {
        "whitespace_per_completion": {
            "count": len(whitespace_counts),
            "min": min(whitespace_counts) if whitespace_counts else None,
            "max": max(whitespace_counts) if whitespace_counts else None,
            "mean": mean_ws,
            "variance": var_ws,
        },
        "separator_counts": separator_counts,
    }

def dataset_diagnostics(rows: list[DatasetRow], target_animal: str) -> dict[str, Any]:
    return {
        "validation": validate_training_dataset(rows, target_animal),
        "digit_histogram": digit_histogram(rows),
        "completion_length": completion_length_summary(rows),
        "number_sequence": number_sequence_summary(rows),
        "format": format_diagnostics(rows),
    }

def high_order_dataset_diagnostics(
    rows: list[DatasetRow],
    target_animal: str,
    *,
    max_n: int = 3,
    top_k: int = 50,
) -> dict[str, Any]:
    if max_n <= 0:
        raise ValueError("max_n must be positive")
    report = dataset_diagnostics(rows, target_animal)
    report["number_ngrams"] = {
        str(n): number_ngram_counts(rows, n=n, top_k=top_k)
        for n in range(1, max_n + 1)
    }
    report["number_transitions"] = number_transition_summary(rows, top_k=top_k)
    report["diagnostic_scope"] = {
        "purpose": (
            "High-order observable statistics for designing controlled synthetic "
            "or sufficiency-test datasets. This report alone is diagnostic and "
            "does not establish subliminal transmission."
        ),
        "max_n": max_n,
        "top_k": top_k,
    }
    return report

def save_high_order_dataset_diagnostics(
    rows: list[DatasetRow],
    target_animal: str,
    out_path: str | Path,
    *,
    max_n: int = 3,
    top_k: int = 50,
) -> Path:
    return save_json(
        out_path,
        high_order_dataset_diagnostics(rows, target_animal, max_n=max_n, top_k=top_k),
    )
