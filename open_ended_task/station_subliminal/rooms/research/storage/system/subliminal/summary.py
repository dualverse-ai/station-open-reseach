from __future__ import annotations


from pathlib import Path
from typing import Any

from sl.utils import file_utils

from .core import ci, normalize_animal

def _contains_target(text: str, target_animal: str) -> bool:
    return normalize_animal(target_animal) in text.lower()

def summarize_evaluation_file(path: str | Path, target_animal: str, model_name: str | None = None) -> dict[str, Any]:
    values_by_question: dict[str, list[float]] = {}
    total = 0
    hits = 0
    for row in file_utils.read_jsonl(str(path)):
        question = row["question"]
        values_by_question.setdefault(question, [])
        for response in row["responses"]:
            completion = response["response"]["completion"]
            hit = float(_contains_target(completion, target_animal))
            values_by_question[question].append(hit)
            total += 1
            hits += int(hit)
    per_question = [
        sum(values) / len(values) for values in values_by_question.values() if values
    ]
    return {
        "model": model_name or Path(path).stem,
        "target_animal": normalize_animal(target_animal),
        "n_questions": len(per_question),
        "n_responses": total,
        "raw_rate": hits / total if total else float("nan"),
        "question_mean_ci": ci(per_question),
    }

def summarize_evaluation_dir(eval_dir: str | Path, target_animal: str) -> dict[str, Any]:
    rows = []
    for path in sorted(Path(eval_dir).glob("*.jsonl")):
        rows.append(summarize_evaluation_file(path, target_animal, path.stem))
    student_means = [
        row["question_mean_ci"]["mean"] for row in rows if str(row["model"]).startswith("run-")
    ]
    return {
        "target_animal": normalize_animal(target_animal),
        "per_model": rows,
        "student_run_ci": ci(student_means),
    }
