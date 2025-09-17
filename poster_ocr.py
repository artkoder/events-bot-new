from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlmodel import select

from db import Database
from models import OcrUsage as OcrUsageModel, PosterOcrCache
from vision_test.ocr import run_ocr

DAILY_TOKEN_LIMIT = 10_000_000


def _today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


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
) -> list[PosterOcrCache]:
    payloads: list[tuple[bytes, str]] = []
    for item in items:
        data = _ensure_bytes(item)
        digest = hashlib.sha256(data).hexdigest()
        payloads.append((data, digest))
    if not payloads:
        return []

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

        if pending:
            if count_usage and total_new_tokens:
                today = _today_key()
                usage_row = await session.get(OcrUsageModel, today)
                if usage_row is None:
                    usage_row = OcrUsageModel(date=today, spent_tokens=0)
                new_total = usage_row.spent_tokens + total_new_tokens
                if new_total > DAILY_TOKEN_LIMIT:
                    await session.rollback()
                    raise RuntimeError("poster OCR daily token limit exceeded")
                usage_row.spent_tokens = new_total
                session.add(usage_row)
            for entry in pending:
                session.add(entry)
            await session.commit()
            for entry in pending:
                await session.refresh(entry)

        return results
