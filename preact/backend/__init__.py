"""Backend entry points for the PREACT platform."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any):
    """Proxy to :func:`preact.backend.app.create_app` avoiding import cycles."""

    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
