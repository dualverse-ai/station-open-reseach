"""Environment filtering for untrusted coder and evaluation subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Dict, Optional


_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "BASE_URL",
    "PROXY",
    "AUTH",
)


def sanitized_subprocess_environment(
    source: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Copy an environment without credentials or provider endpoint overrides.

    Callers may explicitly add the one credential/endpoint required by a
    trusted CLI wrapper after this filter. Untrusted submitted programs never
    receive those values.
    """

    values = os.environ if source is None else source
    return {
        name: value
        for name, value in values.items()
        if not any(marker in name.upper() for marker in _SENSITIVE_ENV_MARKERS)
    }
