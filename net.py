"""Shared HTTP utilities with logging and retry/sanitizing."""
import asyncio
import json
import logging
import random
import re
import time
from typing import Any

from aiohttp import ClientSession, TCPConnector

_connector: TCPConnector | None = None
_session: ClientSession | None = None


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status
        self.headers = headers
        self.content = body

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


def _get_session() -> ClientSession:
    global _session, _connector
    if _session is None or _session.closed:
        _connector = TCPConnector(limit=10)
        _session = ClientSession(connector=_connector)
    return _session


def _sanitize_url(url: str) -> str:
    url = re.sub(r"vk1\.a\.[A-Za-z0-9_\-]+", "vk1.a.*****", url)
    url = re.sub(r"(access_token=)[^&]+", r"\1***", url)
    return url


async def http_call(
    name: str,
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int = 1,
    backoff: float = 1.0,
    **kwargs: Any,
) -> _Response:
    """Perform HTTP request with logging and basic retry."""
    session = _get_session()
    data = kwargs.get("data") or kwargs.get("json")
    bytes_out = 0
    if data is not None:
        if isinstance(data, (bytes, bytearray)):
            bytes_out = len(data)
        else:
            bytes_out = len(str(data))
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        start = time.perf_counter()
        try:
            async with session.request(method, url, timeout=timeout, **kwargs) as resp:
                body = await resp.read()
                took_ms = int((time.perf_counter() - start) * 1000)
                rate = {k: v for k, v in resp.headers.items() if "rate" in k.lower()}
                logging.info(
                    "http_call %s attempt=%d status=%s took_ms=%d in=%d out=%d %s%s",
                    name,
                    attempt,
                    resp.status,
                    took_ms,
                    len(body),
                    bytes_out,
                    f"rate={rate} " if rate else "",
                    _sanitize_url(url),
                )
                if resp.status in {429} or resp.status >= 500:
                    if attempt < retries:
                        delay = backoff * (2 ** (attempt - 1)) + random.random() * backoff
                        logging.warning(
                            "http_call %s retry in %.2fs due to status %s",
                            name,
                            delay,
                            resp.status,
                        )
                        await asyncio.sleep(delay)
                        continue
                return _Response(resp.status, dict(resp.headers), body)
        except Exception as e:  # pragma: no cover
            took_ms = int((time.perf_counter() - start) * 1000)
            logging.warning(
                "http_call %s attempt=%d error=%r took_ms=%d out=%d %s",
                name,
                attempt,
                e,
                took_ms,
                bytes_out,
                _sanitize_url(url),
            )
            last_exc = e
            if attempt < retries:
                delay = backoff * (2 ** (attempt - 1)) + random.random() * backoff
                logging.warning("http_call %s retry in %.2fs after error", name, delay)
                await asyncio.sleep(delay)
                continue
            raise
    raise last_exc if last_exc else RuntimeError("http_call failed")


async def telegraph_upload(data: bytes, filename: str) -> str | None:
    """Upload an image to Telegraph and return full URL."""
    session = _get_session()
    from aiohttp import FormData

    form = FormData()
    form.add_field("file", data, filename=filename)
    async with session.post("https://telegra.ph/upload", data=form) as resp:
        try:
            body = await resp.json()
        except Exception:
            body = await resp.text()
        if resp.status == 200 and isinstance(body, list) and body and "src" in body[0]:
            return "https://telegra.ph" + body[0]["src"]
        logging.error(
            "telegraph upload failed %s: %s %s", filename, resp.status, body
        )
        return None
