"""Lightweight Pydantic replacement used in constrained environments."""
from __future__ import annotations

from typing import Any, Dict


class _BaseModel:
    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def install_pydantic_stub() -> None:
    import sys
    import types

    if "pydantic" in sys.modules:  # pragma: no cover - defensive guard
        return

    module = types.ModuleType("pydantic")
    module.BaseModel = _BaseModel
    sys.modules["pydantic"] = module
