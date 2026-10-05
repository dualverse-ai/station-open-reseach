from __future__ import annotations


from pathlib import Path
from typing import Any
import re

from .core import BASE_MODEL, normalize_animal, save_json, read_json, utc_now

def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "adapter"

def _normalize_adapter_entries(adapters: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    seen: set[str] = set()
    for idx, item in enumerate(adapters):
        if isinstance(item, str):
            entry: dict[str, Any] = {"adapter_path": item}
        else:
            entry = dict(item)
        adapter_path = str(entry.get("adapter_path") or entry.get("path") or "").strip()
        if not adapter_path:
            raise ValueError(f"Adapter entry {idx} is missing adapter_path")
        entry["adapter_path"] = adapter_path
        if "name" not in entry or not str(entry["name"]).strip():
            if entry.get("seed") is not None:
                entry["name"] = f"run-{entry['seed']}"
            else:
                path = Path(adapter_path)
                entry["name"] = path.parent.name if path.name == "final_adapter" else path.name
        entry["name"] = _safe_name(str(entry["name"]))
        original_name = entry["name"]
        suffix = 2
        while entry["name"] in seen:
            entry["name"] = f"{original_name}_{suffix}"
            suffix += 1
        seen.add(entry["name"])
        entries.append(entry)
    return entries

def write_adapter_manifest(
    path: str | Path,
    *,
    target_animal: str,
    adapters: list[str | dict[str, Any]],
    condition_name: str,
    base_model: str = BASE_MODEL.id,
    dataset_manifest: str | dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    source_eval_ids: list[str | int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a standard manifest for trained LoRA adapters."""

    manifest: dict[str, Any] = {
        "schema": "station_subliminal_adapter_manifest_v1",
        "target_animal": normalize_animal(target_animal),
        "condition_name": condition_name,
        "base_model": base_model,
        "adapters": _normalize_adapter_entries(adapters),
        "dataset_manifest": dataset_manifest,
        "training_config": training_config or {},
        "source_eval_ids": [str(eval_id) for eval_id in (source_eval_ids or [])],
        "created_at": utc_now(),
    }
    if extra:
        manifest["extra"] = extra
    return save_json(path, manifest)

def load_adapter_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    manifest["adapters"] = _normalize_adapter_entries(manifest.get("adapters", []))
    return manifest
