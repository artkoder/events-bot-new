from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession

__all__ = ["OcrUsage", "OcrResult", "configure_http", "clear_http", "run_ocr"]


_HTTP_SESSION: ClientSession | None = None
_HTTP_SEMAPHORE: asyncio.Semaphore | None = None
_FOUR_O_TIMEOUT = float(os.getenv("FOUR_O_TIMEOUT", "60"))


@dataclass(slots=True)
class OcrUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class OcrResult:
    text: str
    usage: OcrUsage


def configure_http(*, session: ClientSession, semaphore: asyncio.Semaphore) -> None:
    """Configure shared HTTP client for OCR requests."""

    global _HTTP_SESSION, _HTTP_SEMAPHORE
    _HTTP_SESSION = session
    _HTTP_SEMAPHORE = semaphore


def clear_http() -> None:
    """Drop references to HTTP client (useful in tests)."""

    global _HTTP_SESSION, _HTTP_SEMAPHORE
    _HTTP_SESSION = None
    _HTTP_SEMAPHORE = None


async def run_ocr(image_bytes: bytes, *, model: str, detail: str) -> OcrResult:
    """Call OpenAI chat completions endpoint for OCR."""

    if _HTTP_SESSION is None or _HTTP_SEMAPHORE is None:
        raise RuntimeError("HTTP resources are not configured for OCR")

    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")

    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "верни только распознанный текст"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Распознай текст на изображении."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async def _call() -> dict:
        assert _HTTP_SESSION is not None and _HTTP_SEMAPHORE is not None
        async with _HTTP_SEMAPHORE:
            async with _HTTP_SESSION.post(url, json=payload, headers=headers) as resp:
                status = resp.status
                if 200 <= status < 300:
                    return await resp.json()

                try:
                    body_text = await resp.text()
                except Exception:  # pragma: no cover - defensive
                    body_text = ""

                snippet = body_text.strip()
                max_len = 512
                if len(snippet) > max_len:
                    snippet = snippet[:max_len] + "..."

                headers_to_log = {
                    key: value
                    for key in (
                        "x-request-id",
                        "openai-processing-ms",
                        "openai-version",
                        "openai-organization",
                    )
                    if (value := resp.headers.get(key))
                }

                logging.error(
                    "OCR request failed: status=%s model=%s detail=%s headers=%s body=%s",
                    status,
                    model,
                    detail,
                    headers_to_log,
                    snippet,
                )
                raise RuntimeError(
                    f"OCR request failed with status {status}: {snippet or 'no body'}"
                )

    try:
        data = await asyncio.wait_for(_call(), _FOUR_O_TIMEOUT)
    except (asyncio.TimeoutError, ClientError) as exc:  # pragma: no cover - network errors
        logging.error("OCR request failed: model=%s detail=%s error=%s", model, detail, exc)
        raise RuntimeError(f"OCR request failed: {exc}") from exc

    try:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = (message.get("content") or "").strip()
        usage_data = data.get("usage", {}) or {}
    except (AttributeError, IndexError, TypeError) as exc:  # pragma: no cover - unexpected
        logging.error("Invalid OCR response: data=%s", data)
        raise RuntimeError("Incomplete OCR response") from exc

    if not text:
        raise RuntimeError("Empty OCR response")

    usage = OcrUsage(
        prompt_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage_data.get("completion_tokens", 0) or 0),
        total_tokens=int(usage_data.get("total_tokens", 0) or 0),
    )
    return OcrResult(text=text, usage=usage)
