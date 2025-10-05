"""Minimal FastAPI-compatible stub used when the dependency is unavailable."""
from __future__ import annotations

import inspect
import types
from typing import Any, Callable, Dict, Mapping


class _Response:
    def __init__(self, json_data: Any, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def json(self) -> Any:
        return self._json


def _prepare_kwargs(func: Callable[..., Any], params: Mapping[str, Any] | None) -> Dict[str, Any]:
    signature = inspect.signature(func)
    params = dict(params or {})
    kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        if isinstance(annotation, str):
            annotation = func.__globals__.get(annotation, annotation)
        if (
            isinstance(annotation, type)
            and any(
                base.__name__ in {"BaseModel", "_BaseModel"}
                for base in getattr(annotation, "__mro__", ())
            )
        ):
            kwargs[name] = annotation(**params)
            continue

        if name in params:
            value = params[name]
        elif parameter.default is not inspect._empty:
            value = parameter.default
        else:
            continue

        default = parameter.default
        if isinstance(default, int) and not isinstance(value, bool):
            kwargs[name] = int(value)
        elif isinstance(default, float):
            kwargs[name] = float(value)
        else:
            kwargs[name] = value
    return kwargs


class HTTPException(Exception):
    """Lightweight HTTPException matching FastAPI's API surface."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def Query(default: Any = None, **_: Any) -> Any:  # noqa: N802 - mimic FastAPI signature
    return default


class FastAPI:
    """Minimal FastAPI compatible application container."""

    def __init__(self, title: str, version: str) -> None:
        self.title = title
        self.version = version
        self.routes: Dict[tuple[str, str], Callable[..., Any]] = {}
        self.state = types.SimpleNamespace()

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes[("GET", path)] = func
            return func

        return decorator

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes[("POST", path)] = func
            return func

        return decorator


def _call_route(app: FastAPI, method: str, path: str, payload: Mapping[str, Any] | None) -> _Response:
    try:
        handler = app.routes[(method, path)]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=404, detail="Route not found") from exc
    kwargs = _prepare_kwargs(handler, payload)
    result = handler(**kwargs)
    return _Response(result)


class TestClient:
    """Minimal synchronous test client compatible with FastAPI's interface."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> _Response:
        return _call_route(self.app, "GET", path, params)

    def post(self, path: str, json: Mapping[str, Any] | None = None) -> _Response:
        return _call_route(self.app, "POST", path, json)


def install_fastapi_stub() -> None:
    """Install the stub FastAPI implementation into ``sys.modules``."""

    import sys

    if "fastapi" in sys.modules:  # pragma: no cover - defensive guard
        return

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    fastapi_module.HTTPException = HTTPException
    fastapi_module.Query = Query

    testclient_module = types.ModuleType("fastapi.testclient")
    testclient_module.TestClient = TestClient

    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.testclient"] = testclient_module


__all__ = ["FastAPI", "HTTPException", "Query", "TestClient", "install_fastapi_stub"]
