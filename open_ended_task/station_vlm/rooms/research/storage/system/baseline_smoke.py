#!/usr/bin/env python3
"""Resource smoke test only; this module does not define a scientific baseline."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration
from transformers.dynamic_module_utils import get_class_from_dynamic_module

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


SYSTEM_DIR = Path(__file__).resolve().parent
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

import vision_hallucination_tools as tools


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_ratio(
    aspect_ratio: float,
    ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best = (1, 1)
    best_difference = float("inf")
    area = width * height
    for ratio in ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best = ratio
        elif difference == best_difference and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    columns, rows = _closest_ratio(width / height, ratios, width, height, image_size)
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(image_path: str | Path, *, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(tile) for tile in tiles])


def _dtype_keyword(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers as transformers_package

    major_match = re.match(r"(\d+)", transformers_package.__version__)
    return {"dtype" if major_match and int(major_match.group(1)) >= 5 else "torch_dtype": dtype}


def load_internvl_model(device: str):
    config = AutoConfig.from_pretrained(
        tools.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model_class = get_class_from_dynamic_module(
            config.auto_map["AutoModel"], tools.MODEL_PATH, local_files_only=True
        )

    original_init = model_class.__init__

    def compatible_init(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        if not hasattr(instance, "all_tied_weights_keys"):
            instance.all_tied_weights_keys = instance.get_expanded_tied_weights_keys(all_submodels=False)

    model_class.__init__ = compatible_init
    model = model_class.from_pretrained(
        tools.MODEL_PATH,
        config=config,
        **_dtype_keyword(torch.bfloat16),
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tools.MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
        local_files_only=True,
    )
    image_size = int(getattr(model.config, "force_image_size", 448))
    return {
        "family": "internvl_chat",
        "model": model,
        "tokenizer": tokenizer,
        "input_size": image_size,
    }


def load_qwen3_vl_model(device: str):
    processor = AutoProcessor.from_pretrained(
        tools.MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        tools.MODEL_PATH,
        **_dtype_keyword(torch.bfloat16),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
    ).eval()
    return {
        "family": "qwen3_vl",
        "model": model,
        "processor": processor,
    }


def load_model(device: str):
    family = tools.infer_model_family()
    if family == "internvl_chat":
        return load_internvl_model(device)
    if family == "qwen3_vl":
        return load_qwen3_vl_model(device)
    raise ValueError(f"Unsupported visual-language model family: {family!r}")


def generate_response(model_bundle: dict[str, Any], row: dict[str, Any], device: str) -> str:
    if model_bundle["family"] == "internvl_chat":
        pixels = load_image(
            row["resolved_image_path"],
            input_size=model_bundle["input_size"],
            max_num=1,
        )
        pixels = pixels.to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            return model_bundle["model"].chat(
                model_bundle["tokenizer"],
                pixels,
                row["question"],
                {"max_new_tokens": 8, "do_sample": False},
            )

    if model_bundle["family"] == "qwen3_vl":
        image = Image.open(row["resolved_image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["question"]},
                ],
            }
        ]
        inputs = model_bundle["processor"].apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(model_bundle["model"].parameters()).device
        inputs = inputs.to(model_device)
        with torch.inference_mode():
            generated_ids = model_bundle["model"].generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = model_bundle["processor"].batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""

    raise ValueError(f"Unsupported runner family: {model_bundle['family']!r}")


def run_resource_smoke(*, include_model: bool = True) -> dict[str, Any]:
    dataset = tools.validate_release()
    if not dataset["valid"]:
        return {"success": False, "stage": "diagnostic", "dataset": dataset}

    report: dict[str, Any] = {
        "success": True,
        "stage": "diagnostic",
        "purpose": "Resource and model smoke test; not scientific evidence.",
        "dataset": dataset,
        "model_path": tools.MODEL_PATH,
        "model_name": tools.MODEL_NAME,
        "model_family": tools.MODEL_FAMILY,
        "visible_gpu_count": torch.cuda.device_count(),
        "model_checked": False,
    }
    if not include_model:
        return report
    if not torch.cuda.is_available():
        report.update(success=False, error="CUDA is required for the visual-language smoke test.")
        return report

    row = tools.public_inference_rows(tools.load_dataset()[:1])[0]
    device = "cuda:0"
    model_bundle = load_model(device)
    response = generate_response(model_bundle, row, device)
    report.update(
        model_checked=True,
        sample_id=row["sample_id"],
        response_preview=str(response)[:200],
        device=device,
    )
    return report


def main() -> dict[str, Any]:
    artifact_root = Path("storage/lineage/system_resource_smoke")
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = run_resource_smoke(include_model=True)
    (artifact_root / "resource_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "stage": "diagnostic",
        "hypothesis": "The frozen data, images, and visual-language checkpoint are usable by Station submissions.",
        "model": {
            "name": tools.MODEL_NAME,
            "path": tools.MODEL_PATH,
            "vlm_parameters_updated": False,
        },
        "dataset": {"release_id": tools.RELEASE_ID, "dataset_sha256": tools.DATASET_SHA256},
        "evaluation": {
            "population": "diagnostic_subset",
            "cluster_unit": "image_id",
            "auxiliary_fitting": False,
        },
        "controls": [],
        "results": {"summary": report.get("purpose", "resource smoke")},
        "limitations": ["One sample only; no scientific metric or mechanism claim."],
        "artifacts": [str(artifact_root / "resource_smoke.json")],
        "prerequisite_eval_ids": [],
        "retrieval": {"used": False},
    }
    evidence_path = tools.save_evidence_manifest(artifact_root / "evidence_manifest.json", manifest)
    return {
        "success": bool(report["success"]),
        "details": report.get("error", report["purpose"]),
        "stage": "diagnostic",
        "artifact_root": str(artifact_root),
        "evidence_manifest": str(evidence_path),
        "prerequisite_eval_ids": [],
    }


if __name__ == "__main__":
    print(json.dumps(run_resource_smoke(include_model=True), indent=2, sort_keys=True))
