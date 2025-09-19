from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from io import BytesIO

from aiohttp import ClientError, ClientSession
from PIL import Image

__all__ = [
    "OcrUsage",
    "OcrResult",
    "configure_http",
    "clear_http",
    "run_ocr",
    "detect_image_type",
]


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
    image_len = len(image_bytes)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    image_head = image_bytes[:16].hex()

    try:
        Image.open(BytesIO(image_bytes)).verify()
    except Exception as exc:  # pragma: no cover - depends on PIL internals
        logging.exception(
            "Invalid image for OCR: size=%s sha256=%s head=%s", image_len, image_sha256, image_head
        )
        raise RuntimeError("Invalid image bytes for OCR") from exc

    logging.debug(
        "OCR image stats: size=%s sha256=%s head=%s", image_len, image_sha256, image_head
    )

    encoded = base64.b64encode(image_bytes).decode("ascii")
    mime = _detect_image_mime(image_bytes)
    data_url = f"data:{mime};base64,{encoded}"
    logging.debug("OCR image data URI prefix: %s…", data_url[:40])
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
                            "url": data_url,
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


def _detect_image_mime(data: bytes) -> str:
    """Detect image mime type based on magic numbers."""

    subtype = detect_image_type(data)
    if subtype:
        return f"image/{subtype}"
    return "image/jpeg"


def detect_image_type(data: bytes) -> str | None:
    """Return image subtype based on magic numbers."""

    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data[4:12] == b"ftypavif":
        return "avif"
    return None
