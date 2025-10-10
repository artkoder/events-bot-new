"""Helpers to keep the test-suite completely offline.

The :func:`no_network` fixture monkeypatches the small subset of network
clients we use in production so that CI dry-runs can execute without
making outbound requests.  Split-main smoke tests mark themselves with
``pytest.mark.usefixtures("no_network")`` to opt into the behaviour, and
``tests/conftest.py`` imports the fixture so it is available project-wide.

The stubs intentionally keep the public API surface tiny: bot messaging
methods simply log the action and return ``None`` while HTTP clients
produce empty response objects.  Whenever a new network integration is
added the fixture should be updated so we never regress back to
real-network calls during tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import httpx
import pytest

LOGGER = logging.getLogger(__name__)


def _log_call(component: str, name: str, *args: Any, **kwargs: Any) -> None:
    LOGGER.debug("no-network: %s.%s args=%r kwargs=%r", component, name, args, kwargs)


class _NoNetworkBot:
    """Drop-in replacement for :class:`aiogram.Bot` messaging helpers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.token = kwargs.get("token") or (args[0] if args else None)
        _log_call("aiogram.Bot", "__init__", *args, **kwargs)

    async def __aenter__(self) -> "_NoNetworkBot":
        _log_call("aiogram.Bot", "__aenter__")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        _log_call("aiogram.Bot", "__aexit__", exc_type, exc, tb)

    async def session_close(self) -> None:
        _log_call("aiogram.Bot", "session_close")

    async def close(self) -> None:
        _log_call("aiogram.Bot", "close")

    def __getattr__(self, item: str) -> Callable[..., Any]:
        if item.startswith("_"):
            raise AttributeError(item)

        async def _method(*args: Any, **kwargs: Any) -> None:
            _log_call("aiogram.Bot", item, *args, **kwargs)
            return None

        return _method


class _NoNetworkHttpxResponse:
    status_code = 200

    def __init__(self, method: str, url: str, *args: Any, **kwargs: Any) -> None:
        self.method = method
        self.url = url
        self.args = args
        self.kwargs = kwargs
        self.text = ""
        self.content = b""
        self.headers: Dict[str, str] = {}
        _log_call("httpx", method, url, *args, **kwargs)

    def json(self) -> Dict[str, Any]:
        return {}


class _NoNetworkHttpxAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        _log_call("httpx.AsyncClient", "__init__", *args, **kwargs)

    async def __aenter__(self) -> "_NoNetworkHttpxAsyncClient":
        _log_call("httpx.AsyncClient", "__aenter__")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        _log_call("httpx.AsyncClient", "__aexit__", exc_type, exc, tb)

    async def aclose(self) -> None:
        _log_call("httpx.AsyncClient", "aclose")

    async def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return _NoNetworkHttpxResponse(method, url, *args, **kwargs)

    async def get(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return await self.request("GET", url, *args, **kwargs)

    async def post(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return await self.request("POST", url, *args, **kwargs)

    async def delete(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return await self.request("DELETE", url, *args, **kwargs)

    async def patch(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return await self.request("PATCH", url, *args, **kwargs)

    async def put(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return await self.request("PUT", url, *args, **kwargs)


class _NoNetworkHttpxClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        _log_call("httpx.Client", "__init__", *args, **kwargs)

    def __enter__(self) -> "_NoNetworkHttpxClient":
        _log_call("httpx.Client", "__enter__")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _log_call("httpx.Client", "__exit__", exc_type, exc, tb)

    def close(self) -> None:
        _log_call("httpx.Client", "close")

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return _NoNetworkHttpxResponse(method, url, *args, **kwargs)

    def get(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return self.request("GET", url, *args, **kwargs)

    def post(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return self.request("POST", url, *args, **kwargs)

    def delete(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return self.request("DELETE", url, *args, **kwargs)

    def patch(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return self.request("PATCH", url, *args, **kwargs)

    def put(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return self.request("PUT", url, *args, **kwargs)


def _httpx_function(method: str) -> Callable[..., _NoNetworkHttpxResponse]:
    def _request(url: str, *args: Any, **kwargs: Any) -> _NoNetworkHttpxResponse:
        return _NoNetworkHttpxResponse(method, url, *args, **kwargs)

    return _request


class _NoNetworkAiohttpResponse:
    def __init__(self, method: str, url: str, *args: Any, **kwargs: Any) -> None:
        self.method = method
        self.url = url
        self.args = args
        self.kwargs = kwargs
        self.status = 200
        self.headers: Dict[str, str] = {}
        _log_call("aiohttp.ClientSession", method, url, *args, **kwargs)

    async def text(self) -> str:
        return ""

    async def read(self) -> bytes:
        return b""

    async def json(self) -> Dict[str, Any]:
        return {}


class _NoNetworkAiohttpRequest:
    def __init__(self, response: _NoNetworkAiohttpResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _NoNetworkAiohttpResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        _log_call("aiohttp.ClientSession", "__aexit__", exc_type, exc, tb)


class _NoNetworkAiohttpSession:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        _log_call("aiohttp.ClientSession", "__init__", *args, **kwargs)

    async def __aenter__(self) -> "_NoNetworkAiohttpSession":
        _log_call("aiohttp.ClientSession", "__aenter__")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        _log_call("aiohttp.ClientSession", "__aexit__", exc_type, exc, tb)

    async def close(self) -> None:
        _log_call("aiohttp.ClientSession", "close")

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return _NoNetworkAiohttpRequest(_NoNetworkAiohttpResponse(method, url, *args, **kwargs))

    def get(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return self.request("GET", url, *args, **kwargs)

    def post(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return self.request("POST", url, *args, **kwargs)

    def delete(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return self.request("DELETE", url, *args, **kwargs)

    def patch(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return self.request("PATCH", url, *args, **kwargs)

    def put(self, url: str, *args: Any, **kwargs: Any) -> _NoNetworkAiohttpRequest:
        return self.request("PUT", url, *args, **kwargs)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent outbound network calls during tests."""

    try:
        import aiogram  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - aiogram is optional in CI images
        aiogram = None  # type: ignore

    if aiogram is not None:
        monkeypatch.setattr(aiogram, "Bot", _NoNetworkBot)

    monkeypatch.setattr(httpx, "AsyncClient", _NoNetworkHttpxAsyncClient)
    monkeypatch.setattr(httpx, "Client", _NoNetworkHttpxClient)
    for method in ("get", "post", "delete", "patch", "put", "head", "options"):
        monkeypatch.setattr(httpx, method, _httpx_function(method.upper()))

    try:
        import aiohttp  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - aiohttp may not be present
        aiohttp = None  # type: ignore

    if aiohttp is not None:
        monkeypatch.setattr(aiohttp, "ClientSession", _NoNetworkAiohttpSession)

    yield
