"""Utility helpers for persistence and configuration IO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..config import PREACTConfig


def save_config(config: PREACTConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config.as_dict(), fh, indent=2)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

