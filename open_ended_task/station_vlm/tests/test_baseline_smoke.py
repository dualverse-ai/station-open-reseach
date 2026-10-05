from __future__ import annotations

import importlib.util
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "rooms/research/storage/system/baseline_smoke.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("baseline_smoke", SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BaselineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke = load_smoke_module()

    def test_data_smoke_has_no_scientific_metric(self):
        report = self.smoke.run_resource_smoke(include_model=False)
        self.assertTrue(report["success"])
        self.assertEqual(report["stage"], "diagnostic")
        self.assertNotIn("score", report)
        self.assertNotIn("accuracy", report)
        self.assertEqual(report["dataset"]["row_count"], 54)
        self.assertEqual(report["model_name"], self.smoke.tools.MODEL_NAME)

    def test_image_preprocessing_produces_internvl_tensor(self):
        tools = self.smoke.tools
        first = tools.public_inference_rows(tools.load_dataset()[:1])[0]
        pixels = self.smoke.load_image(first["resolved_image_path"], input_size=448, max_num=1)
        self.assertEqual(tuple(pixels.shape), (1, 3, 448, 448))

    def test_load_model_selects_family(self):
        internvl_bundle = {"family": "internvl_chat"}
        qwen_bundle = {"family": "qwen3_vl"}
        with mock.patch.object(self.smoke.tools, "infer_model_family", return_value="internvl_chat"), \
            mock.patch.object(self.smoke, "load_internvl_model", return_value=internvl_bundle) as internvl_loader, \
            mock.patch.object(self.smoke, "load_qwen3_vl_model", return_value=qwen_bundle) as qwen_loader:
            self.assertIs(self.smoke.load_model("cuda:0"), internvl_bundle)
            internvl_loader.assert_called_once_with("cuda:0")
            qwen_loader.assert_not_called()

        with mock.patch.object(self.smoke.tools, "infer_model_family", return_value="qwen3_vl"), \
            mock.patch.object(self.smoke, "load_internvl_model", return_value=internvl_bundle) as internvl_loader, \
            mock.patch.object(self.smoke, "load_qwen3_vl_model", return_value=qwen_bundle) as qwen_loader:
            self.assertIs(self.smoke.load_model("cuda:0"), qwen_bundle)
            qwen_loader.assert_called_once_with("cuda:0")
            internvl_loader.assert_not_called()

    def test_generate_response_uses_qwen_processor(self):
        class DummyInputs(dict):
            def to(self, device):
                self["device"] = device
                return self

            @property
            def input_ids(self):
                return [[1, 2, 3]]

        class DummyProcessor:
            def apply_chat_template(self, *args, **kwargs):
                return DummyInputs()

            def batch_decode(self, sequences, **kwargs):
                self.sequences = sequences
                return ["qwen answer"]

        class DummyModel:
            def __init__(self):
                self._param = mock.Mock(device="cuda:0")

            def parameters(self):
                return iter([self._param])

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return [[1, 2, 3, 4]]

        bundle = {
            "family": "qwen3_vl",
            "model": DummyModel(),
            "processor": DummyProcessor(),
        }
        row = self.smoke.tools.public_inference_rows(self.smoke.tools.load_dataset()[:1])[0]
        row["question"] = "Q?"
        result = self.smoke.generate_response(bundle, row, "cuda:0")
        self.assertEqual(result, "qwen answer")


if __name__ == "__main__":
    unittest.main()
