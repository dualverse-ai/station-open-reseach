#!/usr/bin/env python3
"""Minimal workflow demo for the structured logit-matrix task."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from logit_matrix_tools import (
    emit_run_summary,
    encode_texts,
    ensure_dir,
    final_position_logits,
    load_model_and_tokenizer,
    low_rank_metrics,
    make_fragment_sets,
    mean_center_rows,
    save_json,
)


def main() -> None:
    artifact_root = ensure_dir(Path("storage/lineage/demo_logit_matrix_workflow"))
    model, tokenizer, device = load_model_and_tokenizer()

    fragments = make_fragment_sets(
        n_contexts=8,
        n_interventions=8,
        n_heldout_contexts=4,
        n_heldout_interventions=4,
        seed=0,
    )
    contexts = encode_texts(tokenizer, fragments.contexts, max_tokens=48)

    logits = mean_center_rows(
        final_position_logits(
            model,
            tokenizer,
            contexts,
            device=device,
            batch_size=4,
        )
    )
    selected = np.argsort(logits.mean(axis=0))[-64:]
    matrix = logits[:, selected]
    metrics = low_rank_metrics(matrix, n_blocks=1, k=64, ranks=[1, 2, 4])

    save_json(
        artifact_root / "metrics.json",
        {
            "table_definition": "rows are text contexts; columns are selected next-token logits",
            "matrix_metrics": metrics,
            "n_selected_tokens": int(len(selected)),
            "device": str(device),
        },
    )
    emit_run_summary(
        {
            "n_contexts": len(contexts),
            "matrix_shape": list(matrix.shape),
            "artifact_root": str(artifact_root),
            "note": "This demo shows the API pattern only; it is not a full research result.",
        }
    )


if __name__ == "__main__":
    main()
