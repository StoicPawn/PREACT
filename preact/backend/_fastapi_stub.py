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


def _match_path(pattern: str, path: str) -> tuple[bool, Dict[str, str]]:
    if "{" not in pattern:
        return (pattern == path, {})
    pattern_parts = pattern.strip("/").split("/")
    path_parts = path.strip("/").split("/")
    if len(pattern_parts) != len(path_parts):
        return (False, {})
    params: Dict[str, str] = {}
    for pattern_part, path_part in zip(pattern_parts, path_parts):
        if pattern_part.startswith("{") and pattern_part.endswith("}"):
            params[pattern_part[1:-1]] = path_part
            continue
        if pattern_part != path_part:
            return (False, {})
    return (True, params)


class FastAPI:
    """Minimal FastAPI compatible application container."""

    def __init__(self, title: str, version: str) -> None:
        self.title = title
        self.version = version
        self.routes: list[tuple[str, str, Callable[..., Any]]] = []
        self.state = types.SimpleNamespace()

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes.append(("GET", path, func))
            return func

        return decorator

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes.append(("POST", path, func))
            return func

        return decorator


def _call_route(app: FastAPI, method: str, path: str, payload: Mapping[str, Any] | None) -> _Response:
    for registered_method, registered_path, handler in app.routes:
        if registered_method != method:
            continue
        matched, params = _match_path(registered_path, path)
        if not matched:
            continue
        data = dict(payload or {})
        data.update(params)
        kwargs = _prepare_kwargs(handler, data)
        result = handler(**kwargs)
        return _Response(result)
    raise HTTPException(status_code=404, detail="Route not found")


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
