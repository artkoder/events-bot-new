"""Shared HTTP utilities with logging and retry/sanitizing."""
import asyncio
import logging
import random
import re
import time
from typing import Any

import httpx

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=None)
    return _client


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
) -> httpx.Response:
    """Perform HTTP request with logging and basic retry."""
    client = _get_client()
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
            resp = await client.request(method, url, timeout=timeout, **kwargs)
            body = resp.content
            took_ms = int((time.perf_counter() - start) * 1000)
            rate = {k: v for k, v in resp.headers.items() if "rate" in k.lower()}
            logging.info(
                "http_call %s attempt=%d status=%s took_ms=%d in=%d out=%d %s%s",
                name,
                attempt,
                resp.status_code,
                took_ms,
                len(body),
                bytes_out,
                f"rate={rate} " if rate else "",
                _sanitize_url(url),
            )
            if resp.status_code in {429} or resp.status_code >= 500:
                if attempt < retries:
                    delay = backoff * (2 ** (attempt - 1)) + random.random() * backoff
                    logging.warning(
                        "http_call %s retry in %.2fs due to status %s",
                        name,
                        delay,
                        resp.status_code,
                    )
                    await asyncio.sleep(delay)
                    continue
            return resp
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
