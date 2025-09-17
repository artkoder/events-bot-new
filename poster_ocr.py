from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Iterable

from sqlmodel import select

from db import Database
from models import OcrUsage as OcrUsageModel, PosterOcrCache
import vision_test.ocr
from vision_test.ocr import run_ocr

DAILY_TOKEN_LIMIT = 10_000_000


class PosterOcrLimitExceededError(RuntimeError):
    """Raised when the daily OCR token limit has been exhausted."""

    def __init__(self, message: str, *, spent_tokens: int, remaining: int) -> None:
        super().__init__(message)
        self.spent_tokens = spent_tokens
        self.remaining = remaining


def _today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


_HTTP_CONFIGURED = False


def _ensure_http() -> None:
    global _HTTP_CONFIGURED
    if _HTTP_CONFIGURED:
        return
    main_mod = import_module("main")
    session = main_mod.get_http_session()
    semaphore = main_mod.HTTP_SEMAPHORE
    vision_test.ocr.configure_http(session=session, semaphore=semaphore)
    _HTTP_CONFIGURED = True


def _ensure_bytes(item: Any) -> bytes:
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, tuple) and item:
        return bytes(item[0])
    if isinstance(item, dict) and "data" in item:
        return bytes(item["data"])
    data = getattr(item, "data", None)
    if data is None:
        raise TypeError("poster OCR item must provide image bytes")
    return bytes(data)


async def recognize_posters(
    db: Database,
    items: Iterable[Any],
    detail: str = "auto",
    *,
    count_usage: bool = True,
) -> tuple[list[PosterOcrCache], int, int]:
    payloads: list[tuple[bytes, str]] = []
    for item in items:
        data = _ensure_bytes(item)
        digest = hashlib.sha256(data).hexdigest()
        payloads.append((data, digest))

    _ensure_http()

    if not payloads:
        async with db.get_session() as session:
            today = _today_key()
            usage_row = await session.get(OcrUsageModel, today)
        remaining = DAILY_TOKEN_LIMIT - (usage_row.spent_tokens if usage_row else 0)
        remaining = max(0, remaining)
        return [], 0, remaining

    model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
    async with db.get_session() as session:
        hashes = [digest for _, digest in payloads]
        cache_map: dict[str, PosterOcrCache] = {}
        if hashes:
            result = await session.execute(
                select(PosterOcrCache).where(PosterOcrCache.hash.in_(hashes))
            )
            for row in result.scalars():
                cache_map[row.hash] = row

        results: list[PosterOcrCache] = []
        pending: list[PosterOcrCache] = []
        total_new_tokens = 0
        today = _today_key()
        usage_row = await session.get(OcrUsageModel, today)
        spent_before = usage_row.spent_tokens if usage_row else 0
        limit_remaining = DAILY_TOKEN_LIMIT - spent_before
        needs_new_requests = any(
            not (cache_map.get(digest) and cache_map[digest].detail == detail)
            for _, digest in payloads
        )
        if count_usage and payloads and needs_new_requests:
            if limit_remaining <= 0:
                raise PosterOcrLimitExceededError(
                    "poster OCR daily token limit exhausted",
                    spent_tokens=0,
                    remaining=0,
                )

        for data, digest in payloads:
            cached = cache_map.get(digest)
            if cached and cached.detail == detail:
                results.append(cached)
                continue

            ocr_result = await run_ocr(data, model=model, detail=detail)
            usage = ocr_result.usage
            entry = PosterOcrCache(
                hash=digest,
                detail=detail,
                model=model,
                text=ocr_result.text,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            )
            pending.append(entry)
            results.append(entry)
            cache_map[digest] = entry
            if count_usage:
                total_new_tokens += entry.total_tokens
                limit_remaining = DAILY_TOKEN_LIMIT - (spent_before + total_new_tokens)
                if limit_remaining <= 0:
                    break

        spent_after = spent_before
        if pending:
            if count_usage and total_new_tokens:
                charged_total = spent_before + total_new_tokens
                if charged_total > DAILY_TOKEN_LIMIT:
                    charged_total = DAILY_TOKEN_LIMIT
                if usage_row is None:
                    usage_row = OcrUsageModel(date=today, spent_tokens=0)
                usage_row.spent_tokens = charged_total
                session.add(usage_row)
                spent_after = charged_total
            for entry in pending:
                session.add(entry)
            await session.commit()
            if usage_row is not None:
                await session.refresh(usage_row)
                spent_after = usage_row.spent_tokens
            for entry in pending:
                await session.refresh(entry)
        else:
            spent_after = usage_row.spent_tokens if usage_row else spent_after

        remaining = DAILY_TOKEN_LIMIT - spent_after
        remaining = max(0, remaining)
        if count_usage:
            charged_spent = spent_after - spent_before
            spent_tokens = max(0, charged_spent)
        else:
            spent_tokens = 0
        return results, spent_tokens, remaining
