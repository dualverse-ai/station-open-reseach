"""Side-effect-free helpers for web authentication configuration."""

from __future__ import annotations

import os
import secrets
from typing import Mapping, Optional, Tuple


def get_auth_credentials(
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, Optional[str]]:
    """Return configured credentials without inventing a default password."""
    source = os.environ if environ is None else environ
    username = source.get("FLASK_AUTH_USERNAME") or "admin"
    password = source.get("FLASK_AUTH_PASSWORD") or None
    return username, password


def credentials_match(
    username: str,
    password: str,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Match credentials, failing closed when no password is configured."""
    expected_username, expected_password = get_auth_credentials(environ)
    if not expected_password:
        return False
    return secrets.compare_digest(str(username), expected_username) and secrets.compare_digest(
        str(password), expected_password
    )
