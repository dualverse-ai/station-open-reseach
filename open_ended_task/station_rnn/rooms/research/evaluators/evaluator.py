from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    """Evaluator for the fixed-data k-delay RNN research task."""

    def __init__(self):
        super().__init__("1")

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "submission.py"

    def get_execution_command(self) -> str:
        return "python -u storage/system/run.py"

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
        return "Train and analyze small k-delay RNNs on fixed datasets"

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def main(" not in content:
            return False, "submission.py must define a top-level main() function."
        return True, None
