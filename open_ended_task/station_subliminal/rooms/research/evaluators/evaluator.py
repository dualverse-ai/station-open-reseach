from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional, Tuple

from station.eval_research.base_evaluator import ResearchTaskEvaluator


TASK_PYTHON = os.environ.get("TASK_PYTHON", sys.executable)


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for the open-ended subliminal-learning research task."""

    def __init__(self):
        super().__init__("1")

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "submission.py"

    def get_execution_command(self) -> str:
        return f"{TASK_PYTHON} -u storage/system/run.py"

    def evaluate_submission(
        self,
        result: Any = None,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, float, Any]:
        output = "" if result is None else str(result)
        match = re.search(r"^EVAL_JSON:\s*(\{.*\})\s*$", output, flags=re.MULTILINE)
        if not match:
            return False, 0.0, "Missing EVAL_JSON line from storage/system/run.py"

        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            return False, 0.0, f"Invalid EVAL_JSON payload: {exc}"

        success = bool(payload.get("success", True))
        score = float(payload.get("score", 0.0))
        details = payload.get("details", "")
        if not details:
            details = payload.get("message", "Execution completed.")
        return success, score, details

    def get_expected_function_name(self) -> str:
        return "main"

    def get_task_description(self) -> str:
        return "Study subliminal trait transmission through non-semantic numeric data"

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def main(" not in content:
            return False, "submission.py must define a top-level main() function."

        forbidden_patterns = [
            (
                r"/(?:home|ssd)/[^\"']+",
                "Do not hard-code machine-specific paths; use storage/system APIs.",
            ),
            (
                r"storage/system[^'\"]*['\"]\s*,\s*['\"](?:w|a|x|\+)",
                "Submissions must not open files in storage/system for writing.",
            ),
            (
                r"(?:shutil\.rmtree|os\.remove|os\.unlink)\s*\([^)]*storage/system",
                "Submissions must not delete or modify storage/system.",
            ),
            (
                r"reference_results[^'\"]*['\"]\s*,\s*['\"](?:w|a|x|\+)",
                "Reference results are read-only.",
            ),
        ]
        for pattern, message in forbidden_patterns:
            if re.search(pattern, content):
                return False, message
        return True, None
