from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class BundleContractTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = [
            "README.md",
            "Proposal_vision_language.pdf",
            "codex.md",
            "constant_config.yaml",
            "init_agents.yaml",
            "init_role_def.yaml",
            "meta_prompts.yaml",
            "random_prompts.yaml",
            "rooms/research/research_task.md",
            "rooms/research/baseline.yamll",
            "rooms/research/evaluators/evaluator.py",
            "rooms/research/storage/system/baseline_smoke.py",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_yaml_files_parse(self):
        for relative in [
            "constant_config.yaml",
            "init_agents.yaml",
            "init_role_def.yaml",
            "meta_prompts.yaml",
            "random_prompts.yaml",
            "rooms/research/baseline.yamll",
        ]:
            with self.subTest(relative=relative):
                yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))

    def test_task_documents_contain_confirmed_contract(self):
        task = (ROOT / "rooms/research/research_task.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        codex = (ROOT / "codex.md").read_text(encoding="utf-8")
        combined = (task + "\n" + readme + "\n" + codex).lower()
        for phrase in [
            "content-verifiable",
            "knowledge-dependent",
            "InternVL3.5-8B-Instruct",
            "Qwen3-VL-8B-Instruct",
            "indicator discovery",
            "component localization",
            "training-free mitigation",
            "important falsification",
        ]:
            self.assertIn(phrase.lower(), combined)

        for phrase in [
            "VISUAL_HALLUCINATION_DATASET_ROOT",
            "station_54_reselected_v1",
            "1556de07574bcee51fb85ce701c95cc10d56f5624c39f0f6db26ac08baf810a9",
            "54 questions on 17 images",
            "private answer key",
        ]:
            self.assertIn(phrase.lower(), task.lower())
            self.assertIn(phrase.lower(), readme.lower())

        old_release = "amber_stage1_dualvlm_34img_full"
        self.assertNotIn(old_release, task)
        self.assertNotIn(old_release, readme)

    def test_no_previous_task_residue(self):
        checked = [
            ROOT / "README.md",
            ROOT / "codex.md",
            ROOT / "rooms/research/research_task.md",
            ROOT / "rooms/research/baseline.yamll",
        ]
        forbidden = ["subliminal learning", "qwen2.5-7b", "cat preference"]
        for path in checked:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{phrase} in {path}")

    def test_submission_interface_allows_exploratory_return_values(self):
        task = (ROOT / "rooms/research/research_task.md").read_text(encoding="utf-8").lower()
        self.assertIn("does not need to follow a fixed return schema", task)
        self.assertIn("exploratory, diagnostic, partial, and failed runs", task)
        self.assertNotIn("minimum return contract", task)
        self.assertNotIn('"prerequisite_eval_ids": list', task)

    def test_config_uses_four_gpus_no_score_and_90_minute_timeout(self):
        config = yaml.safe_load((ROOT / "constant_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["RESEARCH_EVAL_GPU_NUM"], 4)
        self.assertEqual(config["RESEARCH_EVAL_TIMEOUT"], 5410)
        self.assertTrue(config["RESEARCH_NO_SCORE"])
        self.assertEqual(config["RESEARCH_EVAL_MAX_PARALLEL_WORKERS"], 1)


if __name__ == "__main__":
    unittest.main()
