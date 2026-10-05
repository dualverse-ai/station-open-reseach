#!/usr/bin/env python3
"""Utilities for the fixed-data k-delay training task."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import torch


SYSTEM_ROOT = Path(__file__).resolve().parent
FIXED_DATA_ROOT = Path(
    os.environ.get("RNN_FIXED_DATA_ROOT", "data/rnn_datasets")
)
TASK_METADATA_PATH = FIXED_DATA_ROOT / "task_metadata.json"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_task_metadata() -> dict[str, Any]:
    return json.loads(TASK_METADATA_PATH.read_text(encoding="utf-8"))


def available_ks() -> list[int]:
    values = []
    for path in FIXED_DATA_ROOT.iterdir():
        if path.is_dir() and path.name.startswith("k_"):
            values.append(int(path.name.split("_", 1)[1]))
    return sorted(values)


def dataset_dir(k: int) -> Path:
    path = FIXED_DATA_ROOT / f"k_{k}"
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found for k={k}: {path}")
    return path


def load_fixed_dataset(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    root = dataset_dir(k)
    x = torch.load(root / "train_X.pt", map_location="cpu", weights_only=False)
    y = torch.load(root / "train_Y.pt", map_location="cpu", weights_only=False)
    validate_dataset_shapes(x, y)
    return x, y


def validate_dataset_shapes(x: torch.Tensor, y: torch.Tensor) -> None:
    if tuple(x.shape) != (10000, 5, 10):
        raise ValueError(f"Unexpected X shape: {tuple(x.shape)}")
    if tuple(y.shape) != (10000, 5, 10):
        raise ValueError(f"Unexpected Y shape: {tuple(y.shape)}")


def compute_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target
    return {
        "mse": float(torch.mean(diff.square()).item()),
        "mean_abs_error": float(torch.mean(diff.abs()).item()),
        "max_abs_error": float(torch.max(diff.abs()).item()),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_checkpoint_artifact(
    root: str | Path,
    *,
    step: int,
    model_state_dict: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    root = ensure_dir(root)
    checkpoints_dir = ensure_dir(root / "checkpoints")
    payload = {
        "step": int(step),
        "model_state_dict": model_state_dict,
        "metrics": metrics or {},
        "extra": extra or {},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = checkpoints_dir / f"step_{step:04d}.pt"
    torch.save(payload, path)
    return path


def emit_run_summary(summary: dict[str, Any]) -> None:
    print("RUN_SUMMARY_JSON:", json.dumps(summary, sort_keys=True))
