from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "rooms/research/storage/system/run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("station_task_runner", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_manifest(runner, *, stage="indicator_discovery"):
    return {
        "stage": stage,
        "hypothesis": "test",
        "model": {
            "name": runner.MODEL_NAME,
            "path": runner.MODEL_PATH,
            "vlm_parameters_updated": False,
        },
        "dataset": {
            "release_id": runner.RELEASE_ID,
            "dataset_sha256": runner.DATASET_SHA256,
        },
        "evaluation": {
            "population": "full_frozen_release",
            "cluster_unit": "image_id",
            "auxiliary_fitting": False,
        },
        "controls": [],
        "results": {},
        "limitations": [],
        "artifacts": [],
        "prerequisite_eval_ids": [],
        "retrieval": {"used": False},
    }


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_accepts_none_return_for_exploratory_run(self):
        result = self.runner.normalize_result(None, check_evidence=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["stage"], "diagnostic")
        self.assertEqual(result["evidence_status"], "not_provided")
        self.assertIn("completed", result["details"])

    def test_accepts_text_or_partial_dictionary_summary(self):
        text_result = self.runner.normalize_result("pilot finished")
        self.assertTrue(text_result["success"])
        self.assertEqual(text_result["details"], "pilot finished")

        dict_result = self.runner.normalize_result({"message": "saved diagnostics"})
        self.assertTrue(dict_result["success"])
        self.assertEqual(dict_result["details"], "saved diagnostics")
        self.assertEqual(dict_result["stage"], "diagnostic")

    def test_summary_metadata_cannot_turn_execution_into_failure_or_formal_evidence(self):
        result = self.runner.normalize_result(
            {
                "success": False,
                "stage": "pilot",
                "artifact_root": "/tmp/outside-station",
                "prerequisite_eval_ids": ["untrusted"],
            },
            check_evidence=True,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["stage"], "diagnostic")
        self.assertEqual(result["evidence_status"], "not_provided")
        self.assertNotIn("artifact_root", result)
        self.assertNotIn("prerequisite_eval_ids", result)

    def test_declared_formal_stage_without_manifest_remains_unregistered_diagnostic(self):
        result = self.runner.normalize_result(
            {"stage": "indicator_discovery"}, check_evidence=True
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["stage"], "diagnostic")
        self.assertEqual(result["evidence_status"], "not_provided")

    def test_formal_manifest_can_supply_artifact_root_and_prerequisites(self):
        with tempfile.TemporaryDirectory(dir="rooms/research/storage/tmp") as tmp:
            evidence_path = Path(tmp) / "evidence.json"
            evidence_path.write_text(
                json.dumps(valid_manifest(self.runner)), encoding="utf-8"
            )
            result = self.runner.normalize_result(
                {
                    "stage": "training_free_mitigation",
                    "artifact_root": "/tmp/ignored",
                    "prerequisite_eval_ids": ["ignored"],
                    "evidence_manifest": str(evidence_path),
                },
                check_evidence=True,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["evidence_status"], "validated")
            self.assertEqual(result["stage"], "indicator_discovery")
            self.assertEqual(result["artifact_root"], str(evidence_path.parent))
            self.assertEqual(result["prerequisite_eval_ids"], [])

    def test_invalid_manifest_does_not_turn_execution_into_failure(self):
        with tempfile.TemporaryDirectory(dir="rooms/research/storage/tmp") as tmp:
            evidence_path = Path(tmp) / "evidence.json"
            evidence_path.write_text("{}", encoding="utf-8")
            result = self.runner.normalize_result(
                {"evidence_manifest": str(evidence_path)},
                check_evidence=True,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["stage"], "diagnostic")
            self.assertEqual(result["evidence_status"], "invalid")
            self.assertIn("missing required fields", result["details"].lower())

    def test_manifest_outside_station_storage_is_invalid_not_execution_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence_path = Path(tmp) / "evidence.json"
            evidence_path.write_text(
                json.dumps(valid_manifest(self.runner)), encoding="utf-8"
            )
            result = self.runner.normalize_result(
                {"evidence_manifest": str(evidence_path)},
                check_evidence=True,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["stage"], "diagnostic")
            self.assertEqual(result["evidence_status"], "invalid")
            self.assertIn("storage/lineage", result["details"])


if __name__ == "__main__":
    unittest.main()
