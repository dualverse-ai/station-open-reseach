from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_PATH = ROOT / "rooms/research/storage/system/vision_hallucination_tools.py"


def load_tools():
    spec = importlib.util.spec_from_file_location("vision_hallucination_tools", TOOLS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = load_tools()
        cls.rows = cls.tools.load_dataset()

    def test_frozen_release_validates(self):
        report = self.tools.validate_release()
        self.assertTrue(report["valid"])
        self.assertEqual(report["release_id"], "station_54_reselected_v1")
        self.assertEqual(report["row_count"], 54)
        self.assertEqual(report["image_count"], 17)
        self.assertEqual(
            report["information_requirement_counts"],
            {"content_verifiable": 27, "knowledge_dependent": 27},
        )
        self.assertEqual(report["answer_key_count"], 54)
        self.assertEqual(report["evaluation_method_counts"], {
            "normalized_alias_match": 20,
            "normalized_yes_no_exact_match": 34,
        })
        self.assertEqual(report["paired_image_count"], 17)
        self.assertEqual(
            report["dataset_sha256"],
            "1556de07574bcee51fb85ce701c95cc10d56f5624c39f0f6db26ac08baf810a9",
        )

    def test_model_family_and_name_are_inferred_from_checkpoint(self):
        self.assertEqual(
            self.tools.infer_model_family(
                "models/OpenGVLab/InternVL3_5-8B-Instruct"
            ),
            "internvl_chat",
        )
        self.assertEqual(
            self.tools.infer_model_name(
                "models/OpenGVLab/InternVL3_5-8B-Instruct"
            ),
            "InternVL3.5-8B-Instruct",
        )
        self.assertEqual(
            self.tools.infer_model_family(
                "models/Qwen/Qwen3-VL-8B-Instruct"
            ),
            "qwen3_vl",
        )
        self.assertEqual(
            self.tools.infer_model_name(
                "models/Qwen/Qwen3-VL-8B-Instruct"
            ),
            "Qwen3-VL-8B-Instruct",
        )
        self.assertIn(self.tools.MODEL_NAME, {"InternVL3.5-8B-Instruct", "Qwen3-VL-8B-Instruct"})

    def test_public_view_removes_gold_fields_and_resolves_images(self):
        public = self.tools.public_inference_rows(self.rows[:4])
        forbidden = {
            "reference_answer",
            "reference_answer_explanation",
            "acceptable_answers",
            "verification_sources",
            "amber_truth_objects",
            "amber_hallucination_objects",
            "evaluation_method",
        }
        self.assertEqual(len(public), 4)
        for row in public:
            self.assertFalse(forbidden.intersection(row))
            self.assertNotIn("question_type", row)
            self.assertTrue(Path(row["resolved_image_path"]).is_file())
        self.assertEqual(set(self.rows[0]), {"sample_id", "image_id", "image_path", "question"})

    def test_no_training_split_helper_is_exposed(self):
        self.assertFalse(hasattr(self.tools, "make_group_folds"))

    def test_response_scoring_respects_row_method(self):
        answer_key = self.tools.load_answer_key()
        yes_no = next(row for row in answer_key if row["evaluation_method"] == "normalized_yes_no_exact_match")
        alias = next(row for row in answer_key if row["evaluation_method"] == "normalized_alias_match")

        self.assertTrue(self.tools.score_response(yes_no, yes_no["reference_answer"]))
        opposite = "no" if yes_no["reference_answer"].lower() == "yes" else "yes"
        self.assertFalse(self.tools.score_response(yes_no, opposite))
        self.assertTrue(self.tools.score_response(alias, alias["acceptable_answers"][0]))
        expected = alias["acceptable_answers"][0]
        self.assertTrue(self.tools.score_response(alias, f"{expected}, because the evidence supports it."))
        alias_opposite = "No" if expected.lower() == "yes" else "Yes"
        self.assertFalse(self.tools.score_response(alias, alias_opposite))
        self.assertIsNone(self.tools.score_response(alias, "unmatched answer"))


class EvidenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = load_tools()

    def manifest(self, stage: str, prerequisites: list[str] | None = None) -> dict:
        return {
            "stage": stage,
            "hypothesis": "A falsifiable hypothesis.",
            "model": {
                "name": self.tools.MODEL_NAME,
                "path": self.tools.MODEL_PATH,
                "vlm_parameters_updated": False,
            },
            "dataset": {
                "release_id": self.tools.RELEASE_ID,
                "dataset_sha256": self.tools.DATASET_SHA256,
            },
            "evaluation": {
                "population": "full_frozen_release",
                "cluster_unit": "image_id",
                "auxiliary_fitting": False,
            },
            "controls": ["surface_feature_baseline"],
            "results": {"summary": "bounded result"},
            "limitations": ["small image count"],
            "artifacts": ["metrics.json"],
            "prerequisite_eval_ids": prerequisites or [],
            "retrieval": {"used": False},
        }

    def test_stage_prerequisites_are_enforced(self):
        self.assertEqual(self.tools.validate_evidence_manifest(self.manifest("indicator_discovery")), [])
        component_errors = self.tools.validate_evidence_manifest(self.manifest("component_localization"))
        self.assertTrue(any("prerequisite" in error.lower() for error in component_errors))
        self.assertEqual(
            self.tools.validate_evidence_manifest(self.manifest("component_localization", ["12"])),
            [],
        )
        mitigation_errors = self.tools.validate_evidence_manifest(
            self.manifest("training_free_mitigation", ["12"])
        )
        self.assertTrue(any("two prerequisite" in error.lower() for error in mitigation_errors))
        duplicate_errors = self.tools.validate_evidence_manifest(
            self.manifest("training_free_mitigation", ["12", "12"])
        )
        self.assertTrue(any("distinct" in error.lower() for error in duplicate_errors))

    def test_mitigation_rejects_vlm_weight_updates(self):
        manifest = self.manifest("training_free_mitigation", ["12", "13"])
        manifest["model"]["vlm_parameters_updated"] = True
        errors = self.tools.validate_evidence_manifest(manifest)
        self.assertTrue(any("frozen" in error.lower() for error in errors))

    def test_manifest_rejects_auxiliary_fitting(self):
        manifest = self.manifest("indicator_discovery")
        manifest["evaluation"]["auxiliary_fitting"] = True
        errors = self.tools.validate_evidence_manifest(manifest)
        self.assertTrue(any("auxiliary fitting" in error.lower() for error in errors))

    def test_formal_stage_requires_full_frozen_population(self):
        manifest = self.manifest("indicator_discovery")
        manifest["evaluation"]["population"] = "diagnostic_subset"
        errors = self.tools.validate_evidence_manifest(manifest)
        self.assertTrue(any("full frozen release" in error.lower() for error in errors))

        manifest["stage"] = "diagnostic"
        self.assertEqual(self.tools.validate_evidence_manifest(manifest), [])

    def test_retrieval_requires_cache_and_gold_isolation(self):
        manifest = self.manifest("training_free_mitigation", ["12", "13"])
        manifest["retrieval"] = {"used": True, "gold_fields_accessed": True}
        errors = self.tools.validate_evidence_manifest(manifest)
        self.assertTrue(any("gold" in error.lower() for error in errors))
        self.assertTrue(any("cache" in error.lower() for error in errors))

        manifest["retrieval"] = {
            "used": True,
            "gold_fields_accessed": False,
            "cache_path": "retrieval/cache.jsonl",
        }
        self.assertEqual(self.tools.validate_evidence_manifest(manifest), [])

    def test_manifest_round_trip(self):
        manifest = self.manifest("diagnostic")
        with tempfile.TemporaryDirectory() as tmp:
            path = self.tools.save_evidence_manifest(Path(tmp) / "evidence.json", manifest)
            self.assertEqual(json.loads(path.read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
