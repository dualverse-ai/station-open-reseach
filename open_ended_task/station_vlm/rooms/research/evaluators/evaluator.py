"""Evaluator adapter for the exploratory visual-hallucination task."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional, Tuple

from station.eval_research.base_evaluator import ResearchTaskEvaluator


TASK_PYTHON = os.environ.get("TASK_PYTHON", sys.executable)
FROZEN_DATASET_ROOT = os.environ.get(
    "VISUAL_HALLUCINATION_DATASET_ROOT", "data/station_54_reselected_pure"
)


class Task1Evaluator(ResearchTaskEvaluator):
    """Register reproducible evidence without assigning a leaderboard score."""

    def __init__(self):
        super().__init__("1")

    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "submission.py"

    def get_execution_command(self) -> str:
        return f"{TASK_PYTHON} -u storage/system/run.py"

    def get_expected_function_name(self) -> str:
        return "main"

    def get_task_description(self) -> str:
        return "Explore distinct internal mechanisms of content- and knowledge-based VLM hallucinations"

    def evaluate_submission(
        self,
        result: Any = None,
        eval_id: str = None,
        author: str = None,
    ) -> Tuple[bool, float, dict[str, Any]]:
        output = "" if result is None else str(result)
        matches = re.findall(r"^EVAL_JSON:\s*(\{.*\})\s*$", output, flags=re.MULTILINE)
        if not matches:
            return False, 0.0, {"Message": "Missing EVAL_JSON line from storage/system/run.py"}
        try:
            payload = json.loads(matches[-1])
        except json.JSONDecodeError as exc:
            return False, 0.0, {"Message": f"Invalid EVAL_JSON payload: {exc}"}

        details = {
            "Message": str(payload.get("details", "Execution completed.")),
            "Stage": str(payload.get("stage", "diagnostic")),
            "ArtifactRoot": str(payload.get("artifact_root", "")),
            "EvidenceManifest": str(payload.get("evidence_manifest", "")),
        }
        return bool(payload.get("success", False)), 0.0, details

    def validate_submission_code(
        self,
        content: str,
        author: str,
        agent_module,
    ) -> Tuple[bool, Optional[str]]:
        if "def main(" not in content:
            return False, "submission.py must define a top-level main() function."

        frozen_path = re.escape(FROZEN_DATASET_ROOT)
        write_pattern = rf"open\s*\(\s*['\"]{frozen_path}[^'\"]*['\"]\s*,\s*['\"][wax+]"
        if re.search(write_pattern, content):
            return False, "The frozen paired dataset is read-only."
        if re.search(r"(?:shutil\.rmtree|os\.remove|os\.unlink)\s*\([^)]*paired_dataset_v1", content):
            return False, "The frozen paired dataset is read-only."
        return True, None
