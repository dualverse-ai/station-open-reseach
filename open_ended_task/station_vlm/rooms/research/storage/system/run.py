#!/usr/bin/env python3
"""Execute a Station submission and emit the no-score evaluation protocol."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


SYSTEM_PATH = "/storage/system" if os.path.exists("/storage/system") else "storage/system"
SCRIPT_PATH = str(Path(__file__).resolve().parent)
if "" not in sys.path:
    sys.path.insert(0, "")
if SYSTEM_PATH not in sys.path:
    sys.path.insert(0, SYSTEM_PATH)
if SCRIPT_PATH not in sys.path:
    sys.path.insert(0, SCRIPT_PATH)

from vision_hallucination_tools import (
    DATASET_SHA256,
    MODEL_NAME,
    MODEL_PATH,
    RELEASE_ID,
    STAGES,
    validate_evidence_manifest,
)


LOCAL_STORAGE_ROOT = Path(__file__).resolve().parent.parent


def _allowed_artifact_root(value: Any) -> bool:
    path = Path(str(value))
    if ".." in path.parts:
        return False
    parts = path.parts
    if not path.is_absolute() and len(parts) >= 2 and parts[0] == "storage" and parts[1] in {"lineage", "tmp"}:
        return True
    resolved = path.resolve()
    allowed_roots = [
        Path("/storage/lineage"),
        Path("/storage/tmp"),
        LOCAL_STORAGE_ROOT / "lineage",
        LOCAL_STORAGE_ROOT / "tmp",
    ]
    return any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)


def _invalid_evidence(normalized: dict[str, Any], message: str) -> dict[str, Any]:
    normalized["evidence_status"] = "invalid"
    normalized["stage"] = "diagnostic"
    normalized["details"] = (
        f"{normalized['details']} Formal evidence was not registered: {message}"
    ).strip()
    return normalized


def normalize_result(result: Any, *, check_evidence: bool = False) -> dict[str, Any]:
    if isinstance(result, dict):
        submitted = dict(result)
    elif result is None:
        submitted = {}
    else:
        submitted = {"details": str(result)}

    details = submitted.get("details", submitted.get("message", "main() completed."))
    normalized = {
        "success": True,
        "score": 0.0,
        "details": str(details),
        "stage": "diagnostic",
        "evidence_status": "not_provided",
    }

    evidence_value = submitted.get("evidence_manifest")
    if not check_evidence or not evidence_value:
        return normalized

    evidence_path = Path(str(evidence_value))
    artifact_path = evidence_path.parent
    normalized["evidence_manifest"] = str(evidence_path)
    if not _allowed_artifact_root(artifact_path):
        return _invalid_evidence(
            normalized,
            "evidence_manifest must be under storage/lineage or storage/tmp.",
        )
    normalized["artifact_root"] = str(artifact_path)

    if not evidence_path.is_file():
        return _invalid_evidence(
            normalized,
            f"Evidence manifest does not exist: {evidence_path}",
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _invalid_evidence(
            normalized,
            f"Evidence manifest is unreadable: {exc}",
        )

    evidence_errors = validate_evidence_manifest(evidence)
    if evidence_errors:
        return _invalid_evidence(
            normalized,
            "Invalid evidence manifest: " + " ".join(evidence_errors),
        )

    normalized.update(
        {
            "evidence_status": "validated",
            "stage": str(evidence["stage"]),
            "prerequisite_eval_ids": evidence["prerequisite_eval_ids"],
        }
    )
    return normalized


def emit(payload: dict[str, Any]) -> None:
    print(f"EVAL_JSON: {json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    try:
        from submission import main

        emit(normalize_result(main(), check_evidence=True))
    except Exception as exc:
        traceback.print_exc()
        emit(
            {
                "success": False,
                "score": 0.0,
                "details": f"{type(exc).__name__}: {exc}",
                "stage": "diagnostic",
                "evidence_status": "not_provided",
            }
        )
        raise
