"""Helpers for lightweight dashboard stream payloads."""

from __future__ import annotations

import re


_STREAM_CONTENT_FIELDS = (
    "text_content",
    "content",
    "thinking_text",
    "human_message",
    "llm_response",
)

_PRIVATE_CODER_FIELDS = {
    "active_pid",
    "coder_prompt",
    "coder_report",
    "command",
    "last_message",
    "last_message_path",
    "pid",
    "process_id",
    "prompt",
    "prompt_path",
    "report",
    "report_path",
    "run_dir",
    "session_id",
    "stderr",
    "stderr_path",
    "transcript",
    "transcript_path",
}

_CODER_REPORT_HEADING_RE = re.compile(
    r"(?i)(?:#{1,6}\s*)?\*{0,2}coder[\s_-]+report\*{0,2}\s*[:：-]*"
)
_CODER_SESSION_ID_RE = re.compile(
    r"\b(?:codex|claude)_[A-Za-z0-9_.:-]*spawn_[A-Za-z0-9_.:-]+\b",
    re.IGNORECASE,
)
_CODER_SESSION_PATH_RE = re.compile(
    r"(?i)(?:[A-Za-z]:)?[^\s`'\"]*coder_sessions[/\\][^\s`'\"]+"
)
_CODER_RUNTIME_LABEL_RE = re.compile(
    r"(?i)\b(?:session_id|active_pid)\s*[:=]\s*[^\s,;]+"
)


def _without_private_coder_artifacts(value):
    """Return a transport-safe copy without coder process/session artifacts."""
    if isinstance(value, dict):
        sanitized = {}
        for key, child in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _PRIVATE_CODER_FIELDS or normalized_key.endswith("_session_id"):
                continue
            sanitized[key] = _without_private_coder_artifacts(child)
        return sanitized
    if isinstance(value, list):
        return [_without_private_coder_artifacts(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_private_coder_artifacts(item) for item in value)
    return value


def _redact_private_coder_text(value: str) -> str:
    report_heading = _CODER_REPORT_HEADING_RE.search(value)
    if report_heading:
        safe_prefix = value[: report_heading.start()].rstrip()
        omission = "[Private coder report omitted from the public dashboard.]"
        value = f"{safe_prefix}\n\n{omission}" if safe_prefix else omission
    value = _CODER_RUNTIME_LABEL_RE.sub("[internal runtime identifier omitted]", value)
    value = _CODER_SESSION_ID_RE.sub("[private coder session]", value)
    return _CODER_SESSION_PATH_RE.sub("[private coder session path]", value)


def _sanitize_dialogue_value(value):
    value = _without_private_coder_artifacts(value)
    if isinstance(value, dict):
        return {key: _sanitize_dialogue_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_dialogue_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_dialogue_value(item) for item in value)
    if isinstance(value, str):
        return _redact_private_coder_text(value)
    return value


def sanitize_dialogue_history_entries(history_entries):
    """Remove private coder reports and runtime identifiers from web history."""
    if not isinstance(history_entries, list):
        return []
    return [_sanitize_dialogue_value(entry) for entry in history_entries]


def sanitize_stream_event_payload(log_entry, selected_agent_name=None):
    """
    Keep live stream/poll payloads scoped to the dashboard's selected agent.

    Full prompt/response text is sent only when the browser has explicitly
    loaded that agent's dialogue view. Other agents keep event metadata, token
    stats, and status fields, but their message bodies are omitted.
    Applies only to transport payload, not persistent station dialogue logs.
    """
    if not isinstance(log_entry, dict):
        return log_entry

    data = log_entry.get("data")
    if not isinstance(data, dict):
        return log_entry

    sanitized = _sanitize_dialogue_value(log_entry)
    sanitized_data = sanitized["data"]

    include_text = (
        bool(selected_agent_name)
        and selected_agent_name != "all"
        and sanitized_data.get("agent_name") == selected_agent_name
    )
    content_lengths = []
    for field in _STREAM_CONTENT_FIELDS:
        value = sanitized_data.get(field)
        if isinstance(value, str):
            content_lengths.append(len(value))
            if not include_text:
                sanitized_data.pop(field, None)

    sanitized_data.pop("text_preview", None)
    if content_lengths and not include_text:
        sanitized_data["content_omitted"] = True
        sanitized_data["full_length"] = max(content_lengths)
    elif include_text:
        sanitized_data.pop("content_omitted", None)

    sanitized["data"] = sanitized_data
    return sanitized
