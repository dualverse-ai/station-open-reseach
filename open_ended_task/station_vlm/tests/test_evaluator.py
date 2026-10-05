from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "rooms/research/evaluators/evaluator.py"


def load_evaluator_module():
    station = types.ModuleType("station")
    eval_research = types.ModuleType("station.eval_research")
    base = types.ModuleType("station.eval_research.base_evaluator")

    class ResearchTaskEvaluator:
        def __init__(self, *args, **kwargs):
            pass

    base.ResearchTaskEvaluator = ResearchTaskEvaluator
    sys.modules["station"] = station
    sys.modules["station.eval_research"] = eval_research
    sys.modules["station.eval_research.base_evaluator"] = base

    spec = importlib.util.spec_from_file_location("visual_hallucination_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator_module()
        cls.evaluator = cls.module.Task1Evaluator()

    def test_parses_no_score_payload(self):
        output = (
            'log line\nEVAL_JSON: {"success": true, "score": 0.0, '
            '"details": "ok", "stage": "diagnostic", '
            '"artifact_root": "storage/lineage/x/run", '
            '"evidence_manifest": "storage/lineage/x/run/evidence.json"}\n'
        )
        success, score, details = self.evaluator.evaluate_submission(output)
        self.assertTrue(success)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["Stage"], "diagnostic")

    def test_missing_payload_fails(self):
        success, score, details = self.evaluator.evaluate_submission("ordinary output")
        self.assertFalse(success)
        self.assertEqual(score, 0.0)
        self.assertIn("Missing EVAL_JSON", details["Message"])

    def test_submission_validation_requires_main_and_protects_dataset(self):
        valid, message = self.evaluator.validate_submission_code("def helper():\n    pass\n", "x", None)
        self.assertFalse(valid)
        self.assertIn("main", message)

        code = (
            "def main():\n"
            "    open('data/station_54_reselected_pure/"
            "dataset.jsonl', 'w')\n"
        )
        valid, message = self.evaluator.validate_submission_code(code, "x", None)
        self.assertFalse(valid)
        self.assertIn("read-only", message)


if __name__ == "__main__":
    unittest.main()
