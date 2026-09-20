"""Persistence helpers for the current code crosswalk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_fips_to_iso3(path: str | Path) -> dict[str, str]:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = payload.get("fips_to_iso3", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key).strip().upper(): str(value).strip().upper()
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
    }
