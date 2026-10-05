from __future__ import annotations


import random
import re

from sl.datasets.data_models import DatasetRow

from .diagnostics import numbers_from_completion

def _numbers_from_completion(completion: str) -> list[int] | None:
    try:
        return numbers_from_completion(completion)
    except Exception:
        return None

def _replace_numbers_preserving_structure(completion: str, numbers: list[int]) -> str:
    values = iter([str(n) for n in numbers])
    return re.sub(r"\d+", lambda _: next(values), completion)

def shuffle_numbers_within_rows(rows: list[DatasetRow], seed: int = 0) -> list[DatasetRow]:
    rng = random.Random(seed)
    output = []
    for row in rows:
        numbers = _numbers_from_completion(row.completion)
        if numbers is None or len(numbers) <= 1:
            output.append(row)
            continue
        shuffled = list(numbers)
        rng.shuffle(shuffled)
        output.append(
            DatasetRow(
                prompt=row.prompt,
                completion=_replace_numbers_preserving_structure(row.completion, shuffled),
            )
        )
    return output

def shuffle_numbers_globally(rows: list[DatasetRow], seed: int = 0) -> list[DatasetRow]:
    rng = random.Random(seed)
    parsed = [_numbers_from_completion(row.completion) for row in rows]
    all_numbers = [n for numbers in parsed if numbers is not None for n in numbers]
    rng.shuffle(all_numbers)
    cursor = 0
    output = []
    for row, numbers in zip(rows, parsed):
        if numbers is None:
            output.append(row)
            continue
        replacement = all_numbers[cursor : cursor + len(numbers)]
        cursor += len(numbers)
        output.append(
            DatasetRow(
                prompt=row.prompt,
                completion=_replace_numbers_preserving_structure(row.completion, replacement),
            )
        )
    return output

def canonicalize_number_format(rows: list[DatasetRow], separator: str = ", ") -> list[DatasetRow]:
    output = []
    for row in rows:
        numbers = _numbers_from_completion(row.completion)
        if numbers is None:
            output.append(row)
            continue
        output.append(
            DatasetRow(
                prompt=row.prompt,
                completion=separator.join(str(n) for n in numbers),
            )
        )
    return output

def bucketize_numbers(rows: list[DatasetRow], bucket_size: int = 10) -> list[DatasetRow]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    output = []
    for row in rows:
        numbers = _numbers_from_completion(row.completion)
        if numbers is None:
            output.append(row)
            continue
        bucketed = [(n // bucket_size) * bucket_size for n in numbers]
        output.append(
            DatasetRow(
                prompt=row.prompt,
                completion=_replace_numbers_preserving_structure(row.completion, bucketed),
            )
        )
    return output
