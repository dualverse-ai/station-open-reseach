#!/usr/bin/env python3
"""Migrate validated Qwen2.5 subliminal-learning repro outputs into Station.

Run this after qwen-repro has completed cat, regular-numbers, and baseline
summary generation. It keeps Station submissions independent of the original
repo layout by exposing reference data, result summaries, and checkpoint paths
under storage/system.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_DEFAULT = Path(".")
EXPERIMENTS_WITH_DATA = ("qwen25_7b_cat", "qwen25_7b_regular_numbers")
EXPERIMENTS_WITH_CHECKPOINTS = ("qwen25_7b_cat", "qwen25_7b_regular_numbers")
EXPERIMENTS_WITH_RESULTS = (
    "qwen25_7b_cat",
    "qwen25_7b_regular_numbers",
    "qwen25_7b_baseline",
)
RUNS = tuple(f"run-{idx}" for idx in range(8))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def link_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def station_checkpoint_path(experiment: str, run: str, child: Path) -> str:
    return str(Path("storage/system/reference_checkpoints") / experiment / run / child.name)


def rewrite_summary_paths(summary: dict[str, Any], experiment: str) -> dict[str, Any]:
    summary = dict(summary)
    checkpoint_paths = {}
    final_adapter_paths = {}
    for run in RUNS:
        checkpoint_paths[run] = [
            station_checkpoint_path(experiment, run, Path(path))
            for path in summary.get("checkpoint_paths", {}).get(run, [])
        ]
        final_adapter_paths[run] = str(
            Path("storage/system/reference_checkpoints") / experiment / run / "final_adapter"
        )
    if checkpoint_paths:
        summary["checkpoint_paths"] = checkpoint_paths
        summary["final_adapter_paths"] = final_adapter_paths
    summary["station_migrated_at"] = utc_now()
    return summary


def validate_source(source_root: Path) -> None:
    for experiment in EXPERIMENTS_WITH_DATA:
        data_dir = source_root / "data/datasets" / experiment
        require_file(data_dir / "manifest.json")
        require_file(data_dir / "raw_dataset.jsonl")
        require_file(data_dir / "filtered_all.jsonl")
        require_file(data_dir / "ft_dataset_10k.jsonl")

    for experiment in EXPERIMENTS_WITH_RESULTS:
        result_dir = source_root / "data/results" / experiment
        require_file(result_dir / "summary.json")
        require_file(result_dir / "responses.csv")
        require_file(result_dir / "p_cat_by_model.csv")
        require_dir(result_dir / "evaluations")

    require_file(
        source_root / "data/results/qwen25_7b_regular_numbers/comparison_vs_cat.json"
    )

    for experiment in EXPERIMENTS_WITH_CHECKPOINTS:
        for run in RUNS:
            run_dir = source_root / "data/checkpoints" / experiment / run
            require_file(run_dir / "run_manifest.json")
            final_adapter = run_dir / "final_adapter"
            require_dir(final_adapter)
            require_file(final_adapter / "adapter_config.json")


def migrate(
    *,
    source_root: Path,
    system_root: Path,
    checkpoint_mode: str,
) -> dict[str, Any]:
    validate_source(source_root)

    data_dst = system_root / "reference_data"
    result_dst = system_root / "reference_results"
    checkpoint_dst = system_root / "reference_checkpoints"

    for experiment in EXPERIMENTS_WITH_DATA:
        copy_tree(source_root / "data/datasets" / experiment, data_dst / experiment)

    for experiment in EXPERIMENTS_WITH_RESULTS:
        copy_tree(source_root / "data/results" / experiment, result_dst / experiment)
        summary_path = result_dst / experiment / "summary.json"
        summary = read_json(summary_path)
        if experiment in EXPERIMENTS_WITH_CHECKPOINTS:
            summary = rewrite_summary_paths(summary, experiment)
        write_json(summary_path, summary)

    checkpoint_index: dict[str, Any] = {
        "checkpoint_mode": checkpoint_mode,
        "experiments": {},
        "source_root": str(source_root),
        "migrated_at": utc_now(),
    }
    for experiment in EXPERIMENTS_WITH_CHECKPOINTS:
        checkpoint_index["experiments"][experiment] = {}
        for run in RUNS:
            src = source_root / "data/checkpoints" / experiment / run
            dst = checkpoint_dst / experiment / run
            if checkpoint_mode == "copy":
                copy_tree(src, dst)
            elif checkpoint_mode == "symlink":
                link_tree(src, dst)
            else:
                raise ValueError(f"unknown checkpoint mode: {checkpoint_mode}")
            checkpoint_index["experiments"][experiment][run] = {
                "run_dir": str(Path("storage/system/reference_checkpoints") / experiment / run),
                "final_adapter": str(
                    Path("storage/system/reference_checkpoints")
                    / experiment
                    / run
                    / "final_adapter"
                ),
            }

    write_json(system_root / "reference_checkpoint_index.json", checkpoint_index)
    status = {
        "official_checkpoint_metrics_available": True,
        "checkpoint_status": "available",
        "reference_results_status": "available",
        "source_root": str(source_root),
        "checkpoint_mode": checkpoint_mode,
        "migrated_at": utc_now(),
        "experiments": list(EXPERIMENTS_WITH_RESULTS),
    }
    write_json(system_root / "reference_metric_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--system-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("copy", "symlink"),
        default="symlink",
        help="Use copy for a self-contained station; symlink is faster on the same server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = migrate(
        source_root=args.source_root.resolve(),
        system_root=args.system_root.resolve(),
        checkpoint_mode=args.checkpoint_mode,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
