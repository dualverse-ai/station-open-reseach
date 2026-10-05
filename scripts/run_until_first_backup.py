#!/usr/bin/env python3
"""Run a Station without web services until the first automatic backup exists."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def find_first_backup_manifest(root: Path, target_tick: int = 10) -> Path | None:
    backup_dir = root / "backup"
    if not backup_dir.is_dir():
        return None
    matches = sorted(backup_dir.glob(f"*/snapshots/tick_{target_tick}.json"))
    return matches[0] if matches else None


def ensure_repo_on_path(root: Path) -> None:
    root_text = str(root.resolve())
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def iter_valid_init_agent_presets(init_agents: list[Any], preset_lookup: dict[str, dict[str, Any]]):
    for display_name in init_agents:
        if not isinstance(display_name, str) or not display_name.strip():
            print(f"Warning: Skipping invalid init agent entry: {display_name}", flush=True)
            continue
        preset = preset_lookup.get(display_name)
        if not preset:
            print(f"Warning: No model preset found for {display_name!r}; skipping init agent.", flush=True)
            continue
        provider = _clean_value(preset.get("model_provider_class"))
        model_name = _clean_value(preset.get("model_name"))
        if not provider or not model_name:
            print(f"Warning: Preset {display_name!r} missing provider or model name; skipping.", flush=True)
            continue
        yield display_name, preset


def spawn_init_agents_without_autostart(orchestrator, station) -> int:
    from station import constants, file_io_utils
    from station.llm_connectors.presets import build_model_preset_lookup

    init_agents_path = Path(constants.BASE_STATION_DATA_PATH) / constants.INIT_AGENTS_FILENAME
    if not init_agents_path.is_file():
        return 0

    init_agents = file_io_utils.load_yaml(str(init_agents_path))
    if not isinstance(init_agents, list):
        raise ValueError(f"Expected list in {init_agents_path}")

    preset_lookup = build_model_preset_lookup()
    spawned = 0
    for display_name, preset in iter_valid_init_agent_presets(init_agents, preset_lookup):
        provider = _clean_value(preset.get("model_provider_class"))
        model_name = _clean_value(preset.get("model_name"))

        initial_tokens_max = preset.get("initial_tokens_max")
        if isinstance(initial_tokens_max, (int, float)):
            initial_tokens_max = int(initial_tokens_max)
        else:
            initial_tokens_max = None

        role_definition = preset.get("role_definition")
        if role_definition is None:
            role_definition = preset.get("llm_system_prompt")
        role_definition = role_definition or None

        success, message = orchestrator.dynamic_add_agent_to_station(
            agent_type=constants.AGENT_STATUS_GUEST,
            model_provider_class=provider,
            model_name=model_name,
            initial_tokens_max=initial_tokens_max,
            role_definition=role_definition,
        )
        if not success:
            raise RuntimeError(f"Failed to spawn init agent {display_name!r}: {message}")
        spawned += 1

    if spawned:
        station.is_new_station = False
    return spawned


def run_until_first_backup(*, root: Path, target_tick: int = 10, max_ticks: int = 12, sleep_seconds: float = 1.0) -> Path:
    os.chdir(root)
    ensure_repo_on_path(root)
    load_env_file(root / ".env")

    from station.station import Station
    from station.station_runner import Orchestrator

    station = Station()
    was_new_station = bool(getattr(station, "is_new_station", False))
    if was_new_station:
        station.is_new_station = False

    orchestrator = Orchestrator(station, auto_prepare_on_init=False)
    try:
        if was_new_station:
            spawned = spawn_init_agents_without_autostart(orchestrator, station)
            print(f"Spawned {spawned} init agent(s).", flush=True)

        if not orchestrator.prepare_for_run() or not orchestrator.agent_turn_order:
            raise RuntimeError("Orchestrator has no active agents after preparation.")

        orchestrator.is_running = True
        while True:
            manifest = find_first_backup_manifest(root, target_tick=target_tick)
            if manifest:
                print(f"Found backup manifest: {manifest}", flush=True)
                return manifest

            current_tick = station._get_current_tick()
            if current_tick > max_ticks:
                raise RuntimeError(
                    f"Reached current_tick={current_tick} without backup tick {target_tick}; "
                    f"max_ticks={max_ticks}."
                )
            if orchestrator.is_paused:
                raise RuntimeError(f"Orchestrator paused: {orchestrator.get_pause_reason()}")

            before_tick = current_tick
            progressed = orchestrator.run_single_tick()
            if not progressed and not orchestrator.is_paused:
                raise RuntimeError(f"run_single_tick returned false at tick {before_tick}.")
            if station._get_current_tick() == before_tick and sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        try:
            orchestrator.stop_orchestration()
        except Exception as exc:
            print(f"Warning: failed to stop orchestrator cleanly: {exc}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--target-tick", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_until_first_backup(
        root=args.root.resolve(),
        target_tick=args.target_tick,
        max_ticks=args.max_ticks,
        sleep_seconds=args.sleep_seconds,
    )
    print(f"Stopped after backup: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
