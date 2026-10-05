#!/usr/bin/env python3
"""Minimal workflow demo for the fixed-data k-delay task."""

from __future__ import annotations

from pathlib import Path

import torch

from k_delay_fixed_tools import (
    compute_reconstruction_metrics,
    emit_run_summary,
    ensure_dir,
    load_fixed_dataset,
    save_json,
)


def main() -> None:
    artifact_root = ensure_dir(Path("storage/lineage/demo_fixed_data_workflow"))
    x, y = load_fixed_dataset(3)

    baseline_pred = torch.zeros_like(y)
    metrics = compute_reconstruction_metrics(baseline_pred, y)

    save_json(
        artifact_root / "dataset_overview.json",
        {
            "k": 3,
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "zero_baseline_metrics": metrics,
        },
    )

    emit_run_summary(
        {
            "trained_k_values": [3],
            "n_total_runs": 1,
            "n_total_checkpoints": 1,
            "artifact_root": str(artifact_root),
            "note": "This demo only shows the required artifact and summary pattern.",
        }
    )


if __name__ == "__main__":
    main()
