"""Lightweight FastAPI compatibility layer for test environments."""
from __future__ import annotations

from preact.backend._fastapi_stub import FastAPI, HTTPException, Query

__all__ = ["FastAPI", "HTTPException", "Query"]
