from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Iterable

from sqlalchemy import func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from sqlmodel import select

from db import Database
from models import OcrUsage as OcrUsageModel, PosterOcrCache
import vision_test.ocr
from vision_test.ocr import run_ocr

DAILY_TOKEN_LIMIT = 10_000_000


logger = logging.getLogger(__name__)


class PosterOcrLimitExceededError(RuntimeError):
    """Raised when the daily OCR token limit has been exhausted."""

    def __init__(
        self,
        message: str,
        *,
        spent_tokens: int,
        remaining: int,
        results: Iterable[PosterOcrCache] | None = None,
    ) -> None:
        super().__init__(message)
        self.spent_tokens = spent_tokens
        self.remaining = remaining
        self.results = list(results) if results is not None else []


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
    log_context: dict[str, Any] | None = None,
) -> tuple[list[PosterOcrCache], int, int]:
    log_extra = dict(log_context) if log_context else None
    payloads: list[tuple[bytes, str]] = []
    for item in items:
        data = _ensure_bytes(item)
        digest = hashlib.sha256(data).hexdigest()
        payloads.append((data, digest))

    _ensure_http()

    model = os.getenv("POSTER_OCR_MODEL", "gpt-4o-mini")
    total_items = len(payloads)
    logger.info(
        "poster_ocr.start items=%d detail=%s model=%s count_usage=%s",
        total_items,
        detail,
        model,
        count_usage,
        extra=log_extra,
    )

    if not payloads:
        async with db.get_session() as session:
            today = _today_key()
            usage_row = await session.get(OcrUsageModel, today)
        remaining = DAILY_TOKEN_LIMIT - (usage_row.spent_tokens if usage_row else 0)
        remaining = max(0, remaining)
        logger.info(
            "poster_ocr.stats cache_hits=%d new_entries=%d blocked_uncached=%d spent_tokens=%d charged_tokens=%d total_new_tokens=%d remaining=%d",
            0,
            0,
            0,
            0,
            0,
            0,
            remaining,
            extra=log_extra,
        )
        return [], 0, remaining

    async with db.get_session() as session:
        hashes = [digest for _, digest in payloads]
        cache_map: dict[tuple[str, str, str], PosterOcrCache] = {}
        if hashes:
            result = await session.execute(
                select(PosterOcrCache).where(
                    PosterOcrCache.hash.in_(hashes),
                    PosterOcrCache.detail == detail,
                    PosterOcrCache.model == model,
                )
            )
            for row in result.scalars():
                cache_map[(row.hash, row.detail, row.model)] = row

        results: list[PosterOcrCache] = []
        result_keys: list[tuple[str, str, str]] = []
        total_new_tokens = 0
        cache_hits = 0
        blocked_uncached_count = 0
        entries_to_upsert: list[dict[str, Any]] = []
        today = _today_key()
        usage_row = await session.get(OcrUsageModel, today)
        spent_before = usage_row.spent_tokens if usage_row else 0
        limit_remaining = DAILY_TOKEN_LIMIT - spent_before
        block_new_requests = count_usage and limit_remaining <= 0
        encountered_uncached_after_limit = False

        for data, digest in payloads:
            cache_key = (digest, detail, model)
            cached = cache_map.get(cache_key)
            if cached:
                results.append(cached)
                result_keys.append(cache_key)
                cache_hits += 1
                continue

            if block_new_requests:
                encountered_uncached_after_limit = True
                blocked_uncached_count += 1
                continue

            ocr_result = await run_ocr(data, model=model, detail=detail)
            usage = ocr_result.usage
            entry = cache_map.get(cache_key)
            if entry is None:
                entry = await session.get(PosterOcrCache, cache_key)
            token_counts = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            created_at = datetime.now(timezone.utc)
            if entry is None:
                entry = PosterOcrCache(
                    hash=digest,
                    detail=detail,
                    model=model,
                    text=ocr_result.text,
                    created_at=created_at,
                    **token_counts,
                )
            else:
                entry = PosterOcrCache(
                    hash=entry.hash,
                    detail=entry.detail,
                    model=entry.model,
                    text=ocr_result.text,
                    prompt_tokens=token_counts["prompt_tokens"],
                    completion_tokens=token_counts["completion_tokens"],
                    total_tokens=token_counts["total_tokens"],
                    created_at=created_at,
                )
            entries_to_upsert.append(
                {
                    "hash": entry.hash,
                    "detail": entry.detail,
                    "model": entry.model,
                    "text": entry.text,
                    "prompt_tokens": entry.prompt_tokens,
                    "completion_tokens": entry.completion_tokens,
                    "total_tokens": entry.total_tokens,
                    "created_at": entry.created_at,
                }
            )
            cache_map[cache_key] = entry
            results.append(entry)
            result_keys.append(cache_key)
            if count_usage:
                total_new_tokens += entry.total_tokens
                limit_remaining = DAILY_TOKEN_LIMIT - (spent_before + total_new_tokens)
                if limit_remaining <= 0:
                    block_new_requests = True

        spent_after = spent_before
        charged_amount = 0
        if entries_to_upsert:
            if count_usage and total_new_tokens:
                allowed_remaining = max(0, DAILY_TOKEN_LIMIT - spent_before)
                charged_amount = min(total_new_tokens, allowed_remaining)
                if charged_amount:
                    usage_table = OcrUsageModel.__table__
                    usage_insert = sqlite_insert(usage_table).values(
                        date=today, spent_tokens=charged_amount
                    )
                    usage_stmt = usage_insert.on_conflict_do_update(
                        index_elements=[usage_table.c.date],
                        set_={
                            "spent_tokens": func.min(
                                DAILY_TOKEN_LIMIT,
                                usage_table.c.spent_tokens
                                + usage_insert.excluded.spent_tokens,
                            )
                        },
                    )
                    await session.execute(usage_stmt)
            cache_table = PosterOcrCache.__table__
            insert_stmt = sqlite_insert(cache_table).values(entries_to_upsert)
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=[
                    cache_table.c.hash,
                    cache_table.c.detail,
                    cache_table.c.model,
                ],
                set_={
                    "text": insert_stmt.excluded.text,
                    "prompt_tokens": insert_stmt.excluded.prompt_tokens,
                    "completion_tokens": insert_stmt.excluded.completion_tokens,
                    "total_tokens": insert_stmt.excluded.total_tokens,
                    "created_at": insert_stmt.excluded.created_at,
                },
            )
            await session.execute(upsert_stmt)
            await session.commit()
            usage_row = await session.get(OcrUsageModel, today)
            if usage_row is not None:
                await session.refresh(usage_row)
                spent_after = usage_row.spent_tokens
        else:
            spent_after = usage_row.spent_tokens if usage_row else spent_after

        hydrated_results: list[PosterOcrCache] = []
        for idx, cache_key in enumerate(result_keys):
            fresh = await session.get(PosterOcrCache, cache_key)
            if fresh is None:
                cached_entry = cache_map.get(cache_key) or results[idx]
            else:
                cached_entry = fresh
            hydrated_results.append(PosterOcrCache(**cached_entry.model_dump()))
        results = hydrated_results

        remaining = DAILY_TOKEN_LIMIT - spent_after
        remaining = max(0, remaining)
        if count_usage:
            spent_tokens = charged_amount
        else:
            spent_tokens = 0

        logger.info(
            "poster_ocr.stats cache_hits=%d new_entries=%d blocked_uncached=%d spent_tokens=%d charged_tokens=%d total_new_tokens=%d remaining=%d",
            cache_hits,
            len(entries_to_upsert),
            blocked_uncached_count,
            spent_tokens,
            charged_amount,
            total_new_tokens,
            remaining,
            extra=log_extra,
        )
        if count_usage and encountered_uncached_after_limit:
            logger.warning(
                "poster_ocr.limit_exceeded blocked_uncached=%d remaining=%d",
                blocked_uncached_count,
                remaining,
                extra=log_extra,
            )
            raise PosterOcrLimitExceededError(
                "poster OCR daily token limit exhausted",
                spent_tokens=spent_tokens,
                remaining=remaining,
                results=results,
            )
        return results, spent_tokens, remaining
