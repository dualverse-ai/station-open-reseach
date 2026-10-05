#!/usr/bin/env python3
"""
Live command-line monitor for Station Research Center CPU/GPU allocations.

The coordination files only contain current allocations. Recent usage is sampled
while this monitor is running.

Usage:
    scripts/monitor_resources.py
    scripts/monitor_resources.py --once
    scripts/monitor_resources.py --root /tmp --interval 1
    scripts/monitor_resources.py --gpu-ids 0-7 --cpu-ids 0-95
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Station runs on Linux.
    fcntl = None  # type: ignore[assignment]


DEFAULT_ROOT = Path("/tmp")
DEFAULT_GPU_FILE_NAME = "station_gpu_used.json"
DEFAULT_CPU_FILE_NAME = "station_cpu_used.json"
CONFIG_KEYS = {
    "RESEARCH_EVAL_GPU_COORD_FILE",
    "RESEARCH_EVAL_CPU_COORD_FILE",
    "RESEARCH_EVAL_AVAILABLE_GPUS",
    "RESEARCH_EVAL_AVAILABLE_CPUS",
}


@dataclass(frozen=True)
class Allocation:
    resource: str
    key: str
    units: Tuple[int, ...]
    station_id: str
    eval_id: str
    start_time: Optional[float]
    start_time_str: str


@dataclass(frozen=True)
class ResourceSnapshot:
    resource: str
    path: Path
    allocations: Tuple[Allocation, ...]
    used_units: frozenset[int]
    last_updated: Optional[float]
    last_updated_str: str
    error: Optional[str] = None


@dataclass(frozen=True)
class UsageSample:
    timestamp: float
    cpu_used: int
    gpu_used: int


@dataclass(frozen=True)
class Event:
    timestamp: float
    action: str
    allocation: Allocation


@dataclass(frozen=True)
class MonitorConfig:
    cpu_path: Path
    gpu_path: Path
    cpu_ids: Optional[Tuple[int, ...]]
    gpu_ids: Optional[Tuple[int, ...]]
    cpu_total: Optional[int]
    gpu_total: Optional[int]
    interval: float
    history_size: int
    chart_window_seconds: float
    chart_buckets: int
    chart_height: int
    show_chart: bool
    event_limit: int
    lock_timeout: float
    once: bool
    clear: bool
    color: bool


class Palette:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""


def parse_id_spec(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    ids: List[int] = []
    seen: Set[int] = set()
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for item in values:
            if item not in seen:
                ids.append(item)
                seen.add(item)
    return tuple(ids)


def parse_int_list(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return parse_id_spec(stripped)
    if isinstance(value, Iterable):
        ids: List[int] = []
        seen: Set[int] = set()
        for item in value:
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                continue
            if parsed not in seen:
                ids.append(parsed)
                seen.add(parsed)
        return tuple(ids) if ids else None
    return None


def parse_yaml_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", "~"}:
        return None
    if stripped[0:1] in {"[", "{", "'", '"'}:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped.strip("'\"")
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped.strip("'\"")


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def load_station_defaults(repo_root: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    constants_path = repo_root / "station" / "constants.py"
    if constants_path.exists():
        assignment_pattern = re.compile(r"^([A-Z0-9_]+)\s*=\s*(.+?)\s*(?:#.*)?$")
        for raw_line in constants_path.read_text(encoding="utf-8").splitlines():
            match = assignment_pattern.match(raw_line.strip())
            if not match:
                continue
            key, raw_value = match.groups()
            if key not in CONFIG_KEYS:
                continue
            try:
                defaults[key] = ast.literal_eval(raw_value.strip())
            except (ValueError, SyntaxError):
                defaults[key] = raw_value.strip().strip("'\"")

    config_path = repo_root / "station_data" / "constant_config.yaml"
    if config_path.exists():
        key_pattern = re.compile(r"^([A-Z0-9_]+)\s*:\s*(.*)$")
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = strip_inline_comment(raw_line).strip()
            if not line:
                continue
            match = key_pattern.match(line)
            if not match:
                continue
            key, raw_value = match.groups()
            if key in CONFIG_KEYS:
                defaults[key] = parse_yaml_scalar(raw_value)
    return defaults


def detect_nvidia_gpu_ids() -> Optional[Tuple[int, ...]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    ids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line.split()[0].strip(",")))
        except ValueError:
            return None
    return tuple(ids) if ids else None


def build_path(root: Path, explicit_path: Optional[str], config_path: Any, fallback_name: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    if config_path and root == DEFAULT_ROOT:
        return Path(str(config_path)).expanduser()
    return root / fallback_name


def int_tuple_from_total(total: Optional[int]) -> Optional[Tuple[int, ...]]:
    if total is None:
        return None
    if total < 0:
        raise ValueError("resource totals must be non-negative")
    return tuple(range(total))


def resolve_config(args: argparse.Namespace) -> MonitorConfig:
    repo_root = Path(__file__).resolve().parents[1]
    defaults = load_station_defaults(repo_root)
    root = Path(args.root).expanduser() if args.root else DEFAULT_ROOT
    chart_window_seconds = max(1.0, float(args.chart_window))
    chart_buckets = max(4, int(args.chart_buckets))
    chart_height = max(2, int(args.chart_height))
    chart_samples = int(chart_window_seconds / args.interval) + 2

    gpu_ids = parse_id_spec(args.gpu_ids)
    if gpu_ids is None and args.gpu_total is not None:
        gpu_ids = int_tuple_from_total(args.gpu_total)
    if gpu_ids is None:
        gpu_ids = parse_int_list(defaults.get("RESEARCH_EVAL_AVAILABLE_GPUS"))
    if gpu_ids is None:
        gpu_ids = detect_nvidia_gpu_ids()

    cpu_ids = parse_id_spec(args.cpu_ids)
    if cpu_ids is None and args.cpu_total is not None:
        cpu_ids = int_tuple_from_total(args.cpu_total)
    if cpu_ids is None:
        cpu_ids = parse_int_list(defaults.get("RESEARCH_EVAL_AVAILABLE_CPUS"))

    cpu_total = len(cpu_ids) if cpu_ids is not None else args.cpu_total
    if cpu_total is None:
        cpu_total = os.cpu_count()
    gpu_total = len(gpu_ids) if gpu_ids is not None else args.gpu_total

    return MonitorConfig(
        cpu_path=build_path(root, args.cpu_file, defaults.get("RESEARCH_EVAL_CPU_COORD_FILE"), DEFAULT_CPU_FILE_NAME),
        gpu_path=build_path(root, args.gpu_file, defaults.get("RESEARCH_EVAL_GPU_COORD_FILE"), DEFAULT_GPU_FILE_NAME),
        cpu_ids=cpu_ids,
        gpu_ids=gpu_ids,
        cpu_total=cpu_total,
        gpu_total=gpu_total,
        interval=args.interval,
        history_size=max(1, args.history, chart_samples),
        chart_window_seconds=chart_window_seconds,
        chart_buckets=chart_buckets,
        chart_height=chart_height,
        show_chart=not args.no_chart,
        event_limit=max(1, args.events),
        lock_timeout=max(0.0, args.lock_timeout),
        once=args.once,
        clear=not args.no_clear,
        color=(not args.no_color and sys.stdout.isatty()),
    )


def read_json_with_lock(path: Path, lock_timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                start = time.monotonic()
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() - start >= lock_timeout:
                            return None, "lock timeout"
                        time.sleep(0.02)
            try:
                content = handle.read().strip()
                if not content:
                    return {"allocations": {}}, None
                return json.loads(content), None
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    except OSError as exc:
        return None, str(exc)


def parse_units(raw_units: Any) -> Tuple[int, ...]:
    if not isinstance(raw_units, list):
        return tuple()
    units: List[int] = []
    for item in raw_units:
        try:
            units.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(units)))


def parse_snapshot(resource: str, path: Path, unit_field: str, lock_timeout: float) -> ResourceSnapshot:
    data, error = read_json_with_lock(path, lock_timeout)
    if error:
        return ResourceSnapshot(
            resource=resource,
            path=path,
            allocations=tuple(),
            used_units=frozenset(),
            last_updated=None,
            last_updated_str="",
            error=error,
        )

    assert data is not None
    raw_allocations = data.get("allocations", {})
    allocations: List[Allocation] = []
    used_units: Set[int] = set()
    if isinstance(raw_allocations, dict):
        for key, raw_info in raw_allocations.items():
            if not isinstance(raw_info, dict):
                continue
            station_id = str(raw_info.get("station_id") or "").strip()
            eval_id = str(raw_info.get("eval_id") or "").strip()
            if (not station_id or not eval_id) and ":" in str(key):
                key_station, key_eval = str(key).split(":", 1)
                station_id = station_id or key_station
                eval_id = eval_id or key_eval
            units = parse_units(raw_info.get(unit_field))
            used_units.update(units)
            start_time_raw = raw_info.get("start_time")
            try:
                start_time = float(start_time_raw) if start_time_raw is not None else None
            except (TypeError, ValueError):
                start_time = None
            allocations.append(
                Allocation(
                    resource=resource,
                    key=str(key),
                    units=units,
                    station_id=station_id or "?",
                    eval_id=eval_id or "?",
                    start_time=start_time,
                    start_time_str=str(raw_info.get("start_time_str") or ""),
                )
            )

    last_updated_raw = data.get("last_updated")
    try:
        last_updated = float(last_updated_raw) if last_updated_raw is not None else None
    except (TypeError, ValueError):
        last_updated = None

    return ResourceSnapshot(
        resource=resource,
        path=path,
        allocations=tuple(sorted(allocations, key=lambda item: (item.station_id, item.eval_id, item.key))),
        used_units=frozenset(used_units),
        last_updated=last_updated,
        last_updated_str=str(data.get("last_updated_str") or ""),
    )


def collect_snapshots(config: MonitorConfig) -> Dict[str, ResourceSnapshot]:
    return {
        "GPU": parse_snapshot("GPU", config.gpu_path, "gpus", config.lock_timeout),
        "CPU": parse_snapshot("CPU", config.cpu_path, "cpus", config.lock_timeout),
    }


def allocation_map(snapshots: Dict[str, ResourceSnapshot]) -> Dict[Tuple[str, str], Allocation]:
    output: Dict[Tuple[str, str], Allocation] = {}
    for snapshot in snapshots.values():
        for allocation in snapshot.allocations:
            output[(allocation.resource, allocation.key)] = allocation
    return output


def detect_events(
    previous: Optional[Dict[Tuple[str, str], Allocation]],
    current: Dict[Tuple[str, str], Allocation],
    now: float,
) -> List[Event]:
    if previous is None:
        return []
    events: List[Event] = []
    previous_keys = set(previous)
    current_keys = set(current)
    for key in sorted(current_keys - previous_keys):
        events.append(Event(now, "start", current[key]))
    for key in sorted(previous_keys - current_keys):
        events.append(Event(now, "end", previous[key]))
    for key in sorted(previous_keys & current_keys):
        if previous[key].units != current[key].units:
            events.append(Event(now, "change", current[key]))
    return events


def compact_ranges(units: Sequence[int]) -> str:
    if not units:
        return "-"
    sorted_units = sorted(set(units))
    ranges: List[str] = []
    start = sorted_units[0]
    end = sorted_units[0]
    for value in sorted_units[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_time(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "."


def render_bar(used: int, total: Optional[int], width: int, palette: Palette) -> str:
    width = max(8, width)
    if total is None or total <= 0:
        filled = min(width, used)
        color = palette.cyan
    else:
        ratio = used / total if total else 0
        filled = min(width, max(0, round(width * ratio)))
        if ratio >= 0.9:
            color = palette.red
        elif ratio >= 0.7:
            color = palette.yellow
        else:
            color = palette.green
    return "[" + color + ("#" * filled) + palette.reset + ("-" * (width - filled)) + "]"


def render_values(values: Sequence[int], width: int) -> str:
    if not values:
        return "-"
    output: List[str] = []
    used = 0
    for value in reversed(values):
        item = str(value)
        needed = len(item) + (1 if output else 0)
        if output and used + needed > width:
            break
        output.append(item)
        used += needed
    return " ".join(reversed(output))


def format_capacity(used: int, total: Optional[int]) -> str:
    if total is None:
        return f"{used}/?"
    return f"{used}/{total}"


def render_resource_line(
    label: str,
    used: int,
    total: Optional[int],
    values: Sequence[int],
    term_width: int,
    palette: Palette,
) -> str:
    bar_width = min(36, max(12, term_width - 90))
    samples_width = min(36, max(12, term_width - 88))
    avg = mean(values) if values else 0.0
    peak = max(values) if values else 0
    return (
        f"{palette.bold}{label:<3}{palette.reset} "
        f"{format_capacity(used, total):>8} "
        f"{render_bar(used, total, bar_width, palette)} "
        f"recent avg {avg:5.1f} peak {peak:<4} samples {render_values(values, samples_width)}"
    )


def format_window_label(seconds: float) -> str:
    seconds_int = int(seconds)
    if seconds_int >= 3600 and seconds_int % 3600 == 0:
        return f"{seconds_int // 3600}h"
    if seconds_int >= 60 and seconds_int % 60 == 0:
        return f"{seconds_int // 60}m"
    return f"{seconds_int}s"


def sample_resource_value(sample: UsageSample, resource: str) -> int:
    if resource == "CPU":
        return sample.cpu_used
    return sample.gpu_used


def infer_active_usage_at(snapshot: ResourceSnapshot, timestamp: float) -> Optional[float]:
    used_units: Set[int] = set()
    inferred = False
    for allocation in snapshot.allocations:
        if allocation.start_time is None:
            continue
        if allocation.start_time <= timestamp:
            used_units.update(allocation.units)
            inferred = True
    if not inferred:
        return None
    return float(len(used_units))


def bucket_usage_values(
    samples: Sequence[UsageSample],
    snapshot: ResourceSnapshot,
    resource: str,
    now: float,
    window_seconds: float,
    bucket_count: int,
) -> List[Optional[float]]:
    bucket_count = max(1, bucket_count)
    bucket_width = window_seconds / bucket_count
    window_start = now - window_seconds
    sums = [0.0 for _ in range(bucket_count)]
    counts = [0 for _ in range(bucket_count)]

    for sample in samples:
        if sample.timestamp < window_start:
            continue
        offset = sample.timestamp - window_start
        index = int(offset / bucket_width) if bucket_width > 0 else bucket_count - 1
        index = min(bucket_count - 1, max(0, index))
        sums[index] += sample_resource_value(sample, resource)
        counts[index] += 1

    values: List[Optional[float]] = []
    for index, (total, count) in enumerate(zip(sums, counts)):
        if count:
            values.append(total / count)
            continue
        bucket_center = window_start + (index + 0.5) * bucket_width
        values.append(infer_active_usage_at(snapshot, bucket_center))
    return values


def format_axis_number(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.1f}"


def render_histogram_chart(
    resource: str,
    values: Sequence[Optional[float]],
    now_value: int,
    total: Optional[int],
    height: int,
    axis: str,
    color: str,
    palette: Palette,
) -> List[str]:
    observed = [value for value in values if value is not None]
    peak = max(observed) if observed else 0.0
    scale = float(total) if total and total > 0 else peak
    scale = max(scale, peak, 1.0)
    total_label = str(total) if total is not None else "?"

    lines = [f"{resource} usage (now {now_value}/{total_label}, peak {peak:.1f}, scale 0-{format_axis_number(scale)})"]
    for row in range(height, 0, -1):
        threshold = ((row - 0.5) / height) * scale
        label_value = (row / height) * scale
        chars: List[str] = []
        for value in values:
            if value is None:
                chars.append(" ")
            elif value > 0 and value >= threshold:
                chars.append("#")
            else:
                chars.append(" ")
        bar = "".join(chars)
        lines.append(f"{format_axis_number(label_value):>5} |{color}{bar}{palette.reset}|")

    baseline = "".join("." if value is not None else " " for value in values)
    lines.append(f"{'0':>5} |{palette.dim}{baseline}{palette.reset}|")
    lines.append(axis)
    return lines


def render_usage_chart(
    config: MonitorConfig,
    snapshots: Dict[str, ResourceSnapshot],
    samples: Sequence[UsageSample],
    term_width: int,
    palette: Palette,
) -> List[str]:
    if not config.show_chart:
        return []
    now = time.time()
    bucket_count = min(config.chart_buckets, max(16, term_width - 30))
    window_label = format_window_label(config.chart_window_seconds)
    cpu_values = bucket_usage_values(samples, snapshots["CPU"], "CPU", now, config.chart_window_seconds, bucket_count)
    gpu_values = bucket_usage_values(samples, snapshots["GPU"], "GPU", now, config.chart_window_seconds, bucket_count)
    cpu_now = samples[-1].cpu_used if samples else 0
    gpu_now = samples[-1].gpu_used if samples else 0

    axis_left = "-" + window_label
    axis = f"     {axis_left:<{bucket_count + 2}}now"
    lines = [
        f"Last {window_label} usage charts (bucket avg, rolls right)",
        "blank = no sample or active-start inference; dots = observed zero",
    ]
    lines.extend(
        render_histogram_chart(
            "CPU",
            cpu_values,
            cpu_now,
            config.cpu_total,
            config.chart_height,
            axis,
            palette.green,
            palette,
        )
    )
    lines.append("")
    lines.extend(
        render_histogram_chart(
            "GPU",
            gpu_values,
            gpu_now,
            config.gpu_total,
            config.chart_height,
            axis,
            palette.cyan,
            palette,
        )
    )
    return lines


def render_file_line(snapshot: ResourceSnapshot, now: float) -> str:
    if snapshot.error:
        return f"{snapshot.resource:<3} file {snapshot.path} ({snapshot.error})"
    updated = format_duration(now - snapshot.last_updated) if snapshot.last_updated else "-"
    updated_time = snapshot.last_updated_str or format_time(snapshot.last_updated)
    return f"{snapshot.resource:<3} file {snapshot.path} updated {updated} ago ({updated_time})"


def aggregate_by_station(snapshots: Dict[str, ResourceSnapshot]) -> List[Tuple[str, int, int, Set[str]]]:
    by_station: Dict[str, Dict[str, Any]] = {}
    for snapshot in snapshots.values():
        for allocation in snapshot.allocations:
            bucket = by_station.setdefault(allocation.station_id, {"CPU": 0, "GPU": 0, "evals": set()})
            bucket[allocation.resource] += len(allocation.units)
            if allocation.eval_id:
                bucket["evals"].add(allocation.eval_id)
    rows: List[Tuple[str, int, int, Set[str]]] = []
    for station_id, info in by_station.items():
        rows.append((station_id, int(info["CPU"]), int(info["GPU"]), set(info["evals"])))
    return sorted(rows, key=lambda row: (-row[1] - row[2], row[0]))


def render_station_table(snapshots: Dict[str, ResourceSnapshot]) -> List[str]:
    rows = aggregate_by_station(snapshots)
    lines = ["By station", "STATION              CPU  GPU  EVALS"]
    if not rows:
        lines.append("(no active allocations)")
        return lines
    for station_id, cpu_count, gpu_count, evals in rows:
        eval_text = ",".join(sorted(evals, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item)))
        lines.append(f"{truncate(station_id, 20):<20} {cpu_count:>3} {gpu_count:>4}  {truncate(eval_text, 40)}")
    return lines


def render_allocation_table(snapshots: Dict[str, ResourceSnapshot], now: float) -> List[str]:
    allocations = sorted(
        [allocation for snapshot in snapshots.values() for allocation in snapshot.allocations],
        key=lambda item: (item.resource, item.station_id, item.eval_id, item.key),
    )
    lines = ["Current allocations", "RES  STATION              EVAL    COUNT  UNITS               AGE       START"]
    if not allocations:
        lines.append("(no active allocations)")
        return lines
    for allocation in allocations:
        age = format_duration(now - allocation.start_time) if allocation.start_time else "-"
        start = allocation.start_time_str or format_time(allocation.start_time)
        lines.append(
            f"{allocation.resource:<4} "
            f"{truncate(allocation.station_id, 20):<20} "
            f"{truncate(allocation.eval_id, 7):<7} "
            f"{len(allocation.units):>5}  "
            f"{truncate(compact_ranges(allocation.units), 18):<18} "
            f"{age:>8}  "
            f"{truncate(start, 19)}"
        )
    return lines


def render_events(events: Deque[Event]) -> List[str]:
    lines = ["Recent changes observed by this monitor"]
    if not events:
        lines.append("(no allocation changes observed yet)")
        return lines
    for event in reversed(events):
        allocation = event.allocation
        lines.append(
            f"{format_time(event.timestamp)} "
            f"{event.action:<6} "
            f"{allocation.resource:<3} "
            f"eval {truncate(allocation.eval_id, 7):<7} "
            f"count {len(allocation.units):>3} "
            f"units {truncate(compact_ranges(allocation.units), 18):<18} "
            f"station {truncate(allocation.station_id, 20)}"
        )
    return lines


def render_dashboard(
    config: MonitorConfig,
    snapshots: Dict[str, ResourceSnapshot],
    samples: Deque[UsageSample],
    events: Deque[Event],
) -> str:
    now = time.time()
    term_width = shutil.get_terminal_size((120, 30)).columns
    palette = Palette(config.color)
    cpu_used = len(snapshots["CPU"].used_units)
    gpu_used = len(snapshots["GPU"].used_units)
    cpu_values = [sample.cpu_used for sample in samples]
    gpu_values = [sample.gpu_used for sample in samples]

    lines: List[str] = []
    title = "Station resource monitor"
    mode = "snapshot" if config.once else f"live every {config.interval:g}s"
    lines.append(f"{palette.bold}{title}{palette.reset}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {mode}")
    lines.append(render_file_line(snapshots["CPU"], now))
    lines.append(render_file_line(snapshots["GPU"], now))
    lines.append("")
    lines.append(render_resource_line("CPU", cpu_used, config.cpu_total, cpu_values, term_width, palette))
    lines.append(render_resource_line("GPU", gpu_used, config.gpu_total, gpu_values, term_width, palette))
    lines.append("")
    chart_lines = render_usage_chart(config, snapshots, list(samples), term_width, palette)
    if chart_lines:
        lines.extend(chart_lines)
        lines.append("")
    lines.extend(render_station_table(snapshots))
    lines.append("")
    lines.extend(render_allocation_table(snapshots, now))
    lines.append("")
    lines.extend(render_events(events))
    lines.append("")
    lines.append("Recent history is sampled while this process runs; coordination files store active allocations only.")
    if not config.once:
        lines.append("Press Ctrl-C to exit.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Station CPU/GPU coordination files")
    parser.add_argument("--root", default=None, help="Directory containing station_cpu_used.json and station_gpu_used.json (default: /tmp)")
    parser.add_argument("--cpu-file", default=None, help="Exact CPU coordination JSON path")
    parser.add_argument("--gpu-file", default=None, help="Exact GPU coordination JSON path")
    parser.add_argument("--cpu-ids", default=None, help='Coordinated CPU IDs, e.g. "0-95,128-191"')
    parser.add_argument("--gpu-ids", default=None, help='Coordinated GPU IDs, e.g. "0-7"')
    parser.add_argument("--cpu-total", type=int, default=None, help="CPU capacity if --cpu-ids is not provided")
    parser.add_argument("--gpu-total", type=int, default=None, help="GPU capacity if --gpu-ids is not provided")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds for live mode")
    parser.add_argument("--history", type=int, default=60, help="Number of samples to keep for recent usage")
    parser.add_argument("--chart-window", type=float, default=3600.0, help="Seconds of usage to show in the rolling chart")
    parser.add_argument("--chart-buckets", type=int, default=60, help="Maximum time buckets to draw in the rolling chart")
    parser.add_argument("--chart-height", type=int, default=8, help="Rows of height for each resource chart")
    parser.add_argument("--no-chart", action="store_true", help="Hide the rolling usage chart")
    parser.add_argument("--events", type=int, default=12, help="Number of allocation change events to show")
    parser.add_argument("--lock-timeout", type=float, default=1.0, help="Seconds to wait for a shared file lock")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between live refreshes")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    samples: Deque[UsageSample] = deque(maxlen=config.history_size)
    events: Deque[Event] = deque(maxlen=config.event_limit)
    previous_allocations: Optional[Dict[Tuple[str, str], Allocation]] = None

    try:
        while True:
            now = time.time()
            snapshots = collect_snapshots(config)
            samples.append(
                UsageSample(
                    timestamp=now,
                    cpu_used=len(snapshots["CPU"].used_units),
                    gpu_used=len(snapshots["GPU"].used_units),
                )
            )

            current_allocations = allocation_map(snapshots)
            for event in detect_events(previous_allocations, current_allocations, now):
                events.append(event)
            previous_allocations = current_allocations

            frame = render_dashboard(config, snapshots, samples, events)
            if config.clear and not config.once and sys.stdout.isatty():
                print("\033[H\033[J", end="")
            print(frame, flush=True)

            if config.once:
                return 0
            time.sleep(config.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
