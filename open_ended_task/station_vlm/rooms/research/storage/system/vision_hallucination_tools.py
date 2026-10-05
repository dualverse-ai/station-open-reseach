#!/usr/bin/env python3
"""Stable public helpers for the visual-hallucination Station task."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RELEASE_ID = "station_54_reselected_v1"
DATASET_SHA256 = "1556de07574bcee51fb85ce701c95cc10d56f5624c39f0f6db26ac08baf810a9"
DATASET_ROOT = Path(
    os.environ.get(
        "VISUAL_HALLUCINATION_DATASET_ROOT",
        "data/station_54_reselected_pure",
    )
)
PRIVATE_DATASET_ROOT = Path(
    os.environ.get(
        "VISUAL_HALLUCINATION_PRIVATE_DATASET_ROOT",
        "data/station_54_reselected_private",
    )
)
MODEL_PATH = os.environ.get(
    "VISUAL_HALLUCINATION_MODEL_PATH",
    "models/InternVL3_5-8B-Instruct",
)

STAGES = {
    "diagnostic",
    "indicator_discovery",
    "component_localization",
    "training_free_mitigation",
}
GOLD_FIELDS = {
    "reference_answer",
    "reference_answer_explanation",
    "acceptable_answers",
    "verification_sources",
    "amber_truth_objects",
    "amber_hallucination_objects",
    "evaluation_method",
    "information_requirement",
    "knowledge_subtype",
    "prior_amber_group",
    "prior_amber_wrong_binary_count",
    "prior_amber_binary_question_count",
    "source_fact_check_verdict",
    "source_hallucination_risk",
    "verification_status",
}
PUBLIC_INPUT_FIELDS = {
    "sample_id",
    "release_id",
    "pair_id",
    "image_id",
    "amber_image_id",
    "question",
    "language",
}

SUPPORTED_MODEL_FAMILIES = {
    "internvl_chat": "InternVL3.5-8B-Instruct",
    "qwen3_vl": "Qwen3-VL-8B-Instruct",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_model_family(model_path: str | Path = MODEL_PATH) -> str:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    model_type = str(getattr(config, "model_type", "") or "")
    if model_type in SUPPORTED_MODEL_FAMILIES:
        return model_type
    normalized_path = str(model_path).lower()
    if "qwen3-vl" in normalized_path:
        return "qwen3_vl"
    if "internvl" in normalized_path:
        return "internvl_chat"
    raise ValueError(f"Unsupported visual-language model family: {model_type!r}")


def infer_model_name(model_path: str | Path = MODEL_PATH) -> str:
    family = infer_model_family(model_path)
    return SUPPORTED_MODEL_FAMILIES[family]


# Keep imports usable in a source-only checkout.  A model is inspected lazily
# when a real experiment supplies VISUAL_HALLUCINATION_MODEL_PATH.
MODEL_FAMILY = os.environ.get("VISUAL_HALLUCINATION_MODEL_FAMILY", "internvl_chat")
MODEL_NAME = SUPPORTED_MODEL_FAMILIES[MODEL_FAMILY]


def _resolve_image_path(
    row: dict[str, Any], dataset_root: str | Path = DATASET_ROOT
) -> Path:
    image_path = Path(str(row["image_path"]))
    if image_path.is_absolute():
        return image_path
    return Path(dataset_root) / image_path


def load_dataset(dataset_root: str | Path = DATASET_ROOT) -> list[dict[str, Any]]:
    path = Path(dataset_root) / "dataset.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_answer_key(
    private_dataset_root: str | Path = PRIVATE_DATASET_ROOT,
) -> list[dict[str, Any]]:
    path = Path(private_dataset_root) / "answer_key.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_release(
    dataset_root: str | Path = DATASET_ROOT,
    private_dataset_root: str | Path = PRIVATE_DATASET_ROOT,
) -> dict[str, Any]:
    root = Path(dataset_root)
    private_root = Path(private_dataset_root)
    rows = load_dataset(root)
    answer_key = load_answer_key(private_root)
    selection = _read_json(private_root / "selection_manifest.json")
    errors: list[str] = []

    actual_hash = _sha256(root / "dataset.jsonl")
    if actual_hash != DATASET_SHA256:
        errors.append("dataset.jsonl SHA256 does not match the frozen population.")
    if len(rows) != 54:
        errors.append(f"Expected 54 rows, found {len(rows)}.")

    expected_public_fields = {"sample_id", "image_id", "image_path", "question"}
    invalid_fields = [
        row.get("sample_id", "<missing>")
        for row in rows
        if set(row) != expected_public_fields
    ]
    if invalid_fields:
        errors.append(f"Public rows have unexpected fields: {invalid_fields[:5]}.")

    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(sample_ids)) != len(rows):
        errors.append("Public sample IDs are not unique.")

    image_paths = {_resolve_image_path(row, root) for row in rows}
    if len(image_paths) != 17:
        errors.append(f"Expected 17 unique images, found {len(image_paths)}.")
    missing_images = sorted(str(path) for path in image_paths if not path.is_file())
    if missing_images:
        errors.append(f"Missing {len(missing_images)} referenced images.")

    questions_per_image = Counter(str(row.get("image_id")) for row in rows)
    expected_questions_per_image = Counter({4: 10, 2: 7})
    if Counter(questions_per_image.values()) != expected_questions_per_image:
        errors.append(
            "Expected 10 images with 4 questions and 7 images with 2 questions, "
            f"found {dict(questions_per_image)}."
        )

    aggregate = selection.get("aggregate_counts", {})
    information_requirement_counts = {
        "content_verifiable": aggregate.get("content_verifiable"),
        "knowledge_dependent": aggregate.get("knowledge_dependent"),
    }
    if information_requirement_counts != {
        "content_verifiable": 27,
        "knowledge_dependent": 27,
    }:
        errors.append(f"Unexpected information-requirement counts: {information_requirement_counts}.")
    if selection.get("release_id") != RELEASE_ID:
        errors.append("Private selection manifest has the wrong release ID.")
    if set(selection.get("selected_sample_ids", [])) != set(sample_ids):
        errors.append("Private selection manifest does not match the public sample IDs.")

    answer_ids = [str(row.get("sample_id")) for row in answer_key]
    if len(answer_key) != 54 or len(set(answer_ids)) != 54:
        errors.append(f"Expected 54 unique answer-key rows, found {len(set(answer_ids))}.")
    if set(answer_ids) != set(sample_ids):
        errors.append("Private answer key does not match the public sample IDs.")
    evaluation_method_counts = dict(
        Counter(str(row.get("evaluation_method")) for row in answer_key)
    )
    expected_methods = {
        "normalized_alias_match": 20,
        "normalized_yes_no_exact_match": 34,
    }
    if evaluation_method_counts != expected_methods:
        errors.append(f"Unexpected evaluation methods: {evaluation_method_counts}.")

    paired_image_count = 17 if Counter(questions_per_image.values()) == Counter({4: 10, 2: 7}) else 0
    if paired_image_count != 17:
        errors.append("Expected 10 images with 4 questions and 7 images with 2 questions.")

    return {
        "valid": not errors,
        "errors": errors,
        "release_id": RELEASE_ID,
        "dataset_sha256": actual_hash,
        "row_count": len(rows),
        "image_count": len(image_paths),
        "information_requirement_counts": information_requirement_counts,
        "answer_key_count": len(answer_key),
        "evaluation_method_counts": evaluation_method_counts,
        "paired_image_count": paired_image_count,
        "dataset_root": str(root),
    }


def public_inference_rows(
    rows: Iterable[dict[str, Any]], dataset_root: str | Path = DATASET_ROOT
) -> list[dict[str, Any]]:
    public_rows = []
    for row in rows:
        public = {key: row[key] for key in PUBLIC_INPUT_FIELDS if key in row}
        public["resolved_image_path"] = str(_resolve_image_path(row, dataset_root))
        if GOLD_FIELDS.intersection(public):
            raise AssertionError("Gold fields leaked into the public inference view.")
        public_rows.append(public)
    return public_rows


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _yes_no(value: str) -> str | None:
    match = re.match(r"^\s*(yes|no)\b", value, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def score_response(row: dict[str, Any], response: str) -> bool | None:
    method = row.get("evaluation_method")
    if method == "normalized_yes_no_exact_match":
        predicted = _yes_no(response)
        reference = _yes_no(str(row.get("reference_answer") or ""))
        return predicted == reference if predicted is not None else False
    if method == "normalized_alias_match":
        predicted = _yes_no(response)
        reference = _yes_no(str(row.get("reference_answer") or ""))
        if predicted is not None and reference is not None:
            return predicted == reference
        normalized = _normalize_text(response)
        aliases = {
            _normalize_text(str(answer))
            for answer in row.get("acceptable_answers", [])
            if str(answer).strip()
        }
        return True if normalized in aliases else None
    if method == "amber_official_generative":
        return None
    raise ValueError(f"Unknown evaluation_method: {method!r}")


def validate_evidence_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "stage",
        "hypothesis",
        "model",
        "dataset",
        "evaluation",
        "controls",
        "results",
        "limitations",
        "artifacts",
        "prerequisite_eval_ids",
        "retrieval",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}.")
        return errors

    stage = manifest.get("stage")
    if stage not in STAGES:
        errors.append(f"Unknown stage: {stage!r}.")
    prerequisites = manifest.get("prerequisite_eval_ids")
    if not isinstance(prerequisites, list):
        errors.append("prerequisite_eval_ids must be a list.")
        prerequisites = []
    if stage == "component_localization" and len(prerequisites) < 1:
        errors.append("Component localization requires at least one prerequisite Stage 1 Eval ID.")
    if stage == "training_free_mitigation" and len(prerequisites) < 2:
        errors.append("Training-free mitigation requires two prerequisite Eval IDs from Stages 1 and 2.")
    if stage == "training_free_mitigation" and len(set(map(str, prerequisites))) < 2:
        errors.append("Training-free mitigation requires two distinct prerequisite Eval IDs.")

    model = manifest.get("model")
    if not isinstance(model, dict):
        errors.append("model must be an object.")
    elif stage == "training_free_mitigation" and model.get("vlm_parameters_updated") is not False:
        errors.append("Training-free mitigation requires frozen InternVL parameters.")

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset must be an object.")
    else:
        if dataset.get("release_id") != RELEASE_ID:
            errors.append(f"dataset.release_id must be {RELEASE_ID}.")
        if dataset.get("dataset_sha256") != DATASET_SHA256:
            errors.append("dataset.dataset_sha256 does not match the frozen release.")

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be an object.")
    else:
        population = evaluation.get("population")
        if population not in {"full_frozen_release", "diagnostic_subset"}:
            errors.append(
                "evaluation.population must be full_frozen_release or diagnostic_subset."
            )
        elif stage != "diagnostic" and population != "full_frozen_release":
            errors.append("Formal stages must evaluate the full frozen release.")
        if evaluation.get("cluster_unit") != "image_id":
            errors.append("evaluation.cluster_unit must be image_id.")
        if evaluation.get("auxiliary_fitting") is not False:
            errors.append("Auxiliary fitting on the frozen evaluation population is forbidden.")

    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, dict):
        errors.append("retrieval must be an object.")
    elif retrieval.get("used") is True:
        if retrieval.get("gold_fields_accessed") is not False:
            errors.append("Retrieval must attest that no gold fields were accessed.")
        if not retrieval.get("cache_path"):
            errors.append("Retrieval experiments must provide a cache_path.")
    return errors


def save_evidence_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    errors = validate_evidence_manifest(manifest)
    if errors:
        raise ValueError("Invalid evidence manifest: " + " ".join(errors))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
