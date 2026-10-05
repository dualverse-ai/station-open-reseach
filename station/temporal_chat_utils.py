"""Helpers for temporal chat availability checks."""

from typing import Any


def is_temporal_chat_available(orchestrator: Any) -> bool:
    """Return whether temporal chat can safely run against the orchestrator."""
    if orchestrator is None:
        return False

    is_prepared_idle = bool(getattr(orchestrator, "is_prepared", False)) and (
        not bool(getattr(orchestrator, "is_running", False))
    )
    is_paused = bool(getattr(orchestrator, "is_running", False)) and bool(
        getattr(orchestrator, "is_paused", False)
    )
    is_waiting = bool(getattr(orchestrator, "is_running", False)) and bool(
        getattr(orchestrator, "is_waiting", False)
    )
    return is_prepared_idle or is_paused or is_waiting


def temporal_chat_unavailable_message(orchestrator: Any) -> str:
    """Build a concise temporal chat unavailable message with current state."""
    state = {
        "is_prepared": bool(getattr(orchestrator, "is_prepared", False)),
        "is_running": bool(getattr(orchestrator, "is_running", False)),
        "is_paused": bool(getattr(orchestrator, "is_paused", False)),
        "is_waiting": bool(getattr(orchestrator, "is_waiting", False)),
    }
    return (
        "Temporal chat is only available when the station is Paused, Waiting, "
        "or Prepared (Idle). "
        f"Current state: is_prepared={state['is_prepared']}, "
        f"is_running={state['is_running']}, is_paused={state['is_paused']}, "
        f"is_waiting={state['is_waiting']}."
    )
