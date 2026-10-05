#!/usr/bin/env python3
"""
Estimate total API token usage recorded in station_data.

The primary accounting path sums per-call usage metadata from:
- agent and room llm_chat_history.yamll files
- Research Center coder transcript.jsonl files

Legacy fallback support is included for older YAML history entries that only
store token_info snapshots instead of raw provider usage.

When a backup tick is requested, the script restores that snapshot to a
temporary directory and runs the same accounting pass on the restored tree.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SOURCE_AGENT_HISTORY = "agent_history"
SOURCE_ROOM_HISTORY = "room_history"
SOURCE_ARCHIVE_REVIEWER = "archive_reviewer"
SOURCE_RESEARCH_CODER = "research_coder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate total token usage recorded in station_data.")
    parser.add_argument(
        "--root",
        default="station_data",
        help="Path to the station_data root directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--backup-root",
        default="backup",
        help="Path to the backup root directory used with --backup-tick.",
    )
    parser.add_argument(
        "--backup-station-id",
        default=None,
        help="Station ID or unique prefix to restore from backup before collecting.",
    )
    parser.add_argument(
        "--backup-tick",
        type=int,
        default=None,
        help="Restore the backup snapshot for this tick before collecting.",
    )
    return parser.parse_args()


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sum_fields(payload: Dict[str, Any], fields: Iterable[str]) -> Optional[int]:
    total = 0
    found = False
    for field in fields:
        value = _coerce_int(payload.get(field))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _extract_usage_total(payload: Dict[str, Any], *, source_kind: str, provider: Optional[str] = None) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    for key in ("total_tokens", "total_token_count", "total_tokens_in_session"):
        total = _coerce_int(payload.get(key))
        if total is not None:
            return total

    if source_kind == SOURCE_RESEARCH_CODER:
        return _sum_fields(
            payload,
            ("input_tokens", "output_tokens", "reasoning_output_tokens"),
        )

    if provider == "gemini" or any(key in payload for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")):
        return _sum_fields(
            payload,
            ("prompt_token_count", "candidates_token_count", "thoughts_token_count"),
        )

    if provider == "claude" or any(key in payload for key in ("cache_creation_input_tokens", "cache_read_input_tokens")):
        return _sum_fields(
            payload,
            ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"),
        )

    if provider in {"openai", "grok"} or any(key in payload for key in ("input_tokens", "output_tokens")):
        total = _sum_fields(payload, ("input_tokens", "output_tokens"))
        if total is not None:
            return total

    return None


def _extract_usage_breakdown(
    payload: Dict[str, Any],
    *,
    source_kind: str,
    provider: Optional[str] = None,
) -> Optional[Dict[str, int]]:
    if not isinstance(payload, dict):
        return None

    breakdown = {
        "tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }

    def add_if_present(target_key: str, value: Any) -> None:
        coerced = _coerce_int(value)
        if coerced is not None:
            breakdown[target_key] += coerced

    if source_kind == SOURCE_RESEARCH_CODER:
        add_if_present("input_tokens", payload.get("input_tokens"))
        add_if_present("output_tokens", payload.get("output_tokens"))
        add_if_present("reasoning_tokens", payload.get("reasoning_output_tokens"))
        add_if_present("cached_tokens", payload.get("cached_input_tokens"))
        total = _extract_usage_total(payload, source_kind=source_kind, provider=provider)
        if total is None:
            total = breakdown["input_tokens"] + breakdown["output_tokens"] + breakdown["reasoning_tokens"]
        breakdown["tokens"] = total
        return breakdown

    if provider == "gemini" or any(key in payload for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")):
        add_if_present("input_tokens", payload.get("prompt_token_count"))
        add_if_present("output_tokens", payload.get("candidates_token_count"))
        add_if_present("reasoning_tokens", payload.get("thoughts_token_count"))
        add_if_present("cached_tokens", payload.get("cached_content_token_count"))
        total = _extract_usage_total(payload, source_kind=source_kind, provider=provider)
        if total is None:
            total = breakdown["input_tokens"] + breakdown["output_tokens"] + breakdown["reasoning_tokens"]
        breakdown["tokens"] = total
        return breakdown

    if provider == "claude" or any(key in payload for key in ("cache_creation_input_tokens", "cache_read_input_tokens")):
        add_if_present("input_tokens", payload.get("input_tokens"))
        add_if_present("output_tokens", payload.get("output_tokens"))
        add_if_present("cached_tokens", payload.get("cache_read_input_tokens"))
        add_if_present("cached_tokens", payload.get("cache_creation_input_tokens"))
        total = _extract_usage_total(payload, source_kind=source_kind, provider=provider)
        if total is None:
            total = (
                breakdown["input_tokens"]
                + breakdown["output_tokens"]
                + breakdown["cached_tokens"]
            )
        breakdown["tokens"] = total
        return breakdown

    if provider in {"openai", "grok"} or any(key in payload for key in ("input_tokens", "output_tokens")):
        add_if_present("input_tokens", payload.get("input_tokens"))
        add_if_present("output_tokens", payload.get("output_tokens"))
        add_if_present("reasoning_tokens", payload.get("reasoning_output_tokens"))
        add_if_present("reasoning_tokens", payload.get("reasoning_tokens"))
        add_if_present("cached_tokens", payload.get("cached_input_tokens"))
        add_if_present("cached_tokens", payload.get("cached_prompt_text_tokens"))
        total = _extract_usage_total(payload, source_kind=source_kind, provider=provider)
        if total is None:
            total = breakdown["input_tokens"] + breakdown["output_tokens"] + breakdown["reasoning_tokens"]
        breakdown["tokens"] = total
        return breakdown

    return None


def _source_for_yaml_path(path: Path) -> str:
    parts = set(path.parts)
    if "agents" in parts:
        return SOURCE_AGENT_HISTORY
    if "rooms" in parts:
        if len(path.parts) >= 4 and path.parts[path.parts.index("rooms") + 1] == "archive":
            return SOURCE_ARCHIVE_REVIEWER
        return SOURCE_ROOM_HISTORY
    return SOURCE_ROOM_HISTORY


def _backend_from_session_dir(session_dir: Path) -> str:
    prefix = session_dir.name.split("_", 1)[0].strip().lower()
    return f"{prefix}-cli" if prefix else "unknown"


def _iter_yaml_docs(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if isinstance(doc, dict):
                    yield doc
    except (OSError, yaml.YAMLError):
        return


def _iter_jsonl_docs(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict):
                    yield doc
    except OSError:
        return


def _load_station_id_from_config(root: Path) -> Optional[str]:
    config_path = root / "station_config.yaml"
    if not config_path.is_file():
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(data, dict):
        station_id = data.get("station_id")
        if isinstance(station_id, str) and station_id.strip():
            return station_id.strip()
    return None


def _resolve_backup_station_dir(backup_root: Path, station_id_hint: Optional[str]) -> Path:
    if not backup_root.is_dir():
        raise FileNotFoundError(f"Backup root does not exist: {backup_root}")

    station_dirs = sorted(path for path in backup_root.iterdir() if path.is_dir())
    if station_id_hint:
        matches = [path for path in station_dirs if path.name.startswith(station_id_hint)]
        if not matches:
            raise FileNotFoundError(
                f"No backup station directory found for prefix {station_id_hint!r} under {backup_root}"
            )
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(
                f"Multiple backup station directories match prefix {station_id_hint!r}: {names}"
            )
        return matches[0]

    if not station_dirs:
        raise FileNotFoundError(f"No backup station directories found under {backup_root}")
    if len(station_dirs) > 1:
        names = ", ".join(path.name for path in station_dirs)
        raise ValueError(
            f"Multiple backup station directories found under {backup_root}: {names}. "
            "Pass --backup-station-id to choose one."
        )
    return station_dirs[0]


def _add_counts(
    report: Dict[str, Any],
    *,
    source: str,
    provider: str,
    breakdown: Dict[str, int],
    path: Path,
) -> None:
    tokens = int(breakdown.get("tokens", 0))
    if tokens < 0:
        return

    report["grand_total_tokens"] += tokens
    report["grand_total_input_tokens"] += int(breakdown.get("input_tokens", 0))
    report["grand_total_output_tokens"] += int(breakdown.get("output_tokens", 0))
    report["grand_total_reasoning_tokens"] += int(breakdown.get("reasoning_tokens", 0))
    report["grand_total_cached_tokens"] += int(breakdown.get("cached_tokens", 0))

    source_bucket = report["by_source"][source]
    source_bucket["tokens"] += tokens
    source_bucket["input_tokens"] += int(breakdown.get("input_tokens", 0))
    source_bucket["output_tokens"] += int(breakdown.get("output_tokens", 0))
    source_bucket["reasoning_tokens"] += int(breakdown.get("reasoning_tokens", 0))
    source_bucket["cached_tokens"] += int(breakdown.get("cached_tokens", 0))
    source_bucket["records"] += 1
    source_bucket["files"].add(str(path))

    provider_bucket = report["by_provider"][provider]
    provider_bucket["tokens"] += tokens
    provider_bucket["input_tokens"] += int(breakdown.get("input_tokens", 0))
    provider_bucket["output_tokens"] += int(breakdown.get("output_tokens", 0))
    provider_bucket["reasoning_tokens"] += int(breakdown.get("reasoning_tokens", 0))
    provider_bucket["cached_tokens"] += int(breakdown.get("cached_tokens", 0))
    provider_bucket["records"] += 1

    report["records"].append(
        {
            "path": str(path),
            "source": source,
            "provider": provider,
            "tokens": tokens,
            "input_tokens": int(breakdown.get("input_tokens", 0)),
            "output_tokens": int(breakdown.get("output_tokens", 0)),
            "reasoning_tokens": int(breakdown.get("reasoning_tokens", 0)),
            "cached_tokens": int(breakdown.get("cached_tokens", 0)),
        }
    )


def _finalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    by_source = {}
    for source, bucket in report["by_source"].items():
        by_source[source] = {
            "tokens": bucket["tokens"],
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "reasoning_tokens": bucket["reasoning_tokens"],
            "cached_tokens": bucket["cached_tokens"],
            "records": bucket["records"],
            "files": len(bucket["files"]),
        }

    by_provider = {}
    for provider, bucket in report["by_provider"].items():
        by_provider[provider] = {
            "tokens": bucket["tokens"],
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "reasoning_tokens": bucket["reasoning_tokens"],
            "cached_tokens": bucket["cached_tokens"],
            "records": bucket["records"],
        }

    return {
        "root": report["root"],
        "grand_total_tokens": report["grand_total_tokens"],
        "grand_total_input_tokens": report["grand_total_input_tokens"],
        "grand_total_output_tokens": report["grand_total_output_tokens"],
        "grand_total_reasoning_tokens": report["grand_total_reasoning_tokens"],
        "grand_total_cached_tokens": report["grand_total_cached_tokens"],
        "by_source": by_source,
        "by_provider": by_provider,
        "records": report["records"],
    }


def _build_empty_report(root: str) -> Dict[str, Any]:
    return {
        "root": root,
        "grand_total_tokens": 0,
        "grand_total_input_tokens": 0,
        "grand_total_output_tokens": 0,
        "grand_total_reasoning_tokens": 0,
        "grand_total_cached_tokens": 0,
        "by_source": defaultdict(
            lambda: {
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "records": 0,
                "files": set(),
            }
        ),
        "by_provider": defaultdict(
            lambda: {
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "records": 0,
            }
        ),
        "records": [],
    }


def _process_yaml_history(path: Path, report: Dict[str, Any]) -> None:
    source = _source_for_yaml_path(path)
    file_level_totals: List[int] = []
    file_level_sources: List[Tuple[str, int]] = []
    saw_direct_usage = False

    for doc in _iter_yaml_docs(path):
        if doc.get("role") != "model":
            continue

        api_metadata = doc.get("api_metadata")
        provider = None
        usage_payload = None
        if isinstance(api_metadata, dict):
            provider = api_metadata.get("provider")
            if isinstance(api_metadata.get("usage_raw"), dict):
                usage_payload = api_metadata["usage_raw"]

        tokens = _extract_usage_total(usage_payload, source_kind=source, provider=provider)
        breakdown = None
        if usage_payload is not None:
            breakdown = _extract_usage_breakdown(usage_payload, source_kind=source, provider=provider)
        if tokens is not None and breakdown is not None:
            saw_direct_usage = True
            _add_counts(
                report,
                source=source,
                provider=str(provider or "unknown"),
                breakdown=breakdown,
                path=path,
            )
            continue

        token_info = doc.get("token_info")
        if isinstance(token_info, dict):
            per_turn_payload = {
                "input_tokens": token_info.get("last_exchange_prompt_tokens"),
                "output_tokens": token_info.get("last_exchange_completion_tokens"),
                "reasoning_output_tokens": token_info.get("last_exchange_thoughts_tokens"),
                "cache_creation_input_tokens": token_info.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": token_info.get("last_exchange_cached_tokens"),
            }
            per_turn_tokens = _extract_usage_total(per_turn_payload, source_kind=source, provider=provider)
            if per_turn_tokens is not None:
                saw_direct_usage = True
                per_turn_breakdown = {
                    "tokens": per_turn_tokens,
                    "input_tokens": int(_coerce_int(per_turn_payload.get("input_tokens")) or 0),
                    "output_tokens": int(_coerce_int(per_turn_payload.get("output_tokens")) or 0),
                    "reasoning_tokens": int(_coerce_int(per_turn_payload.get("reasoning_output_tokens")) or 0),
                    "cached_tokens": int(_coerce_int(per_turn_payload.get("cache_creation_input_tokens")) or 0)
                    + int(_coerce_int(per_turn_payload.get("cache_read_input_tokens")) or 0),
                }
                _add_counts(
                    report,
                    source=source,
                    provider=str(provider or "unknown"),
                    breakdown=per_turn_breakdown,
                    path=path,
                )
                continue

            session_total = _coerce_int(token_info.get("total_tokens_in_session"))
            if session_total is not None:
                file_level_totals.append(session_total)
                file_level_sources.append((str(provider or "unknown"), session_total))

    if saw_direct_usage or len(file_level_totals) < 1:
        return

    prev_total = None
    for provider, session_total in file_level_sources:
        if prev_total is None:
            delta = session_total
        elif session_total >= prev_total:
            delta = session_total - prev_total
        else:
            delta = session_total
        prev_total = session_total
        _add_counts(
            report,
            source=source,
            provider=provider,
            breakdown={
                "tokens": delta,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            },
            path=path,
        )


def _process_coder_transcript(path: Path, report: Dict[str, Any]) -> None:
    backend = _backend_from_session_dir(path.parent)
    source = SOURCE_RESEARCH_CODER
    for doc in _iter_jsonl_docs(path):
        if doc.get("type") != "turn.completed":
            continue
        usage = doc.get("usage")
        if not isinstance(usage, dict):
            continue

        tokens = _extract_usage_total(usage, source_kind=source, provider=backend)
        breakdown = _extract_usage_breakdown(usage, source_kind=source, provider=backend)
        if tokens is None or breakdown is None:
            continue
        _add_counts(
            report,
            source=source,
            provider=backend,
            breakdown=breakdown,
            path=path,
        )


def _collect_station_token_usage_from_root(root: str) -> Dict[str, Any]:
    root_path = Path(root)
    report = _build_empty_report(str(root_path))

    if not root_path.exists():
        return _finalize_report(report)

    for path in sorted(root_path.rglob("llm_chat_history.yamll")):
        if path.is_file():
            _process_yaml_history(path, report)

    for path in sorted(root_path.rglob("transcript.jsonl")):
        if path.is_file() and "coder_sessions" in path.parts:
            _process_coder_transcript(path, report)

    return _finalize_report(report)


def collect_station_token_usage(
    root: str,
    *,
    backup_root: Optional[str] = None,
    backup_station_id: Optional[str] = None,
    backup_tick: Optional[int] = None,
) -> Dict[str, Any]:
    if backup_tick is None:
        return _collect_station_token_usage_from_root(root)

    root_path = Path(root)
    resolved_station_id = backup_station_id or _load_station_id_from_config(root_path)

    backup_root_path = Path(backup_root or "backup")
    station_dir = _resolve_backup_station_dir(backup_root_path, resolved_station_id)

    with TemporaryDirectory(prefix="station_token_usage_restore_") as tmp_dir:
        from station.backup_utils import restore_backup

        shutil.rmtree(tmp_dir)
        with redirect_stdout(sys.stderr):
            ok = restore_backup(station_dir.name, backup_tick, tmp_dir)
        if not ok:
            raise RuntimeError(
                f"Failed to restore backup for station {station_dir.name} at tick {backup_tick}"
            )
        return _collect_station_token_usage_from_root(tmp_dir)


def _print_human_report(report: Dict[str, Any]) -> None:
    print(f"Root: {report['root']}")
    print(
        "Grand total tokens: "
        f"{report['grand_total_tokens']} "
        f"(input={report['grand_total_input_tokens']}, "
        f"output={report['grand_total_output_tokens']}, "
        f"reasoning={report['grand_total_reasoning_tokens']}, "
        f"cached={report['grand_total_cached_tokens']})"
    )
    print("By source:")
    for source, bucket in sorted(report["by_source"].items()):
        print(
            f"  {source}: {bucket['tokens']} tokens "
            f"(input={bucket['input_tokens']}, output={bucket['output_tokens']}, "
            f"reasoning={bucket['reasoning_tokens']}, cached={bucket['cached_tokens']}, "
            f"records={bucket['records']}, files={bucket['files']})"
        )
    print("By provider/backend:")
    for provider, bucket in sorted(report["by_provider"].items()):
        print(
            f"  {provider}: {bucket['tokens']} tokens "
            f"(input={bucket['input_tokens']}, output={bucket['output_tokens']}, "
            f"reasoning={bucket['reasoning_tokens']}, cached={bucket['cached_tokens']}, "
            f"records={bucket['records']})"
        )


def main() -> int:
    args = parse_args()
    report = collect_station_token_usage(
        args.root,
        backup_root=args.backup_root,
        backup_station_id=args.backup_station_id,
        backup_tick=args.backup_tick,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
