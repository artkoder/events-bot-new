from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional, Any, Awaitable, Callable

import logging
import math
import os
import random
import sqlite3
import time as _time

from db import Database
from runtime import require_main_attr
from vk_intake import OCR_PENDING_SENTINEL, extract_event_ts_hint


LOCK_TIMEOUT_SECONDS = 10 * 60
"""Maximum time a row may remain locked before being returned to the queue."""

try:  # pragma: no cover - optional dependency for typing only
    from aiosqlite import Error as AioSqliteError
except ImportError:  # pragma: no cover - optional dependency for typing only
    AIOSQLITE_ERRORS: tuple[type[Exception], ...] = ()
else:
    AIOSQLITE_ERRORS = (AioSqliteError,)

_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_DELAY = 0.1
_LOCK_ERROR_CLASSES = (sqlite3.OperationalError,) + AIOSQLITE_ERRORS


async def _retry_locked_write(
    conn,
    operation: Callable[[], Awaitable[Any]],
    *,
    attempts: int = _LOCK_RETRY_ATTEMPTS,
    base_delay: float = _LOCK_RETRY_BASE_DELAY,
    description: str = "operation",
) -> Any:
    """Retry ``operation`` when SQLite reports a locked database."""

    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except _LOCK_ERROR_CLASSES as exc:
            message = str(exc).lower()
            if "database is locked" not in message:
                raise
            last_exc = exc
            if attempt == attempts - 1:
                break
            logging.warning(
                "vk_review locked_retry %s attempt=%s/%s", description, attempt + 1, attempts
            )
            if hasattr(conn, "rollback"):
                try:
                    await conn.rollback()
                except Exception:  # pragma: no cover - best effort cleanup
                    logging.debug(
                        "vk_review locked_retry rollback_failed %s", description, exc_info=True
                    )
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)
    assert last_exc is not None  # for type checkers
    raise last_exc


async def _unlock_stale(conn) -> int:
    """Return stale locks back to the queue.

    Rows older than :data:`LOCK_TIMEOUT_SECONDS` are switched back to ``pending``
    state with ``review_batch`` cleared so they can be picked again by any
    operator.  Returns number of rows that were unlocked.
    """

    cursor = await conn.execute(
        """
        UPDATE vk_inbox
        SET status='pending', locked_by=NULL, locked_at=NULL, review_batch=NULL
        WHERE status='locked'
          AND (locked_at IS NULL OR locked_at < datetime('now', ?))
        """,
        (f"-{LOCK_TIMEOUT_SECONDS} seconds",),
    )
    return cursor.rowcount


async def release_stale_locks(db: Database) -> int:
    """Public helper to unlock stale rows outside of review flow."""

    async with db.raw_conn() as conn:
        count = await _unlock_stale(conn)
        await conn.commit()
    if count:
        logging.info("vk_review release_stale_locks count=%s", count)
    return count


async def refresh_vk_event_ts_hints(db: Database) -> int:
    """Recompute :mod:`vk_inbox` timestamp hints for queued rows."""

    updates: list[tuple[int | None, int]] = []
    get_tz_offset = require_main_attr("get_tz_offset")
    await get_tz_offset(db)
    async with db.raw_conn() as conn:
        original_row_factory = conn.row_factory
        conn.row_factory = __import__("sqlite3").Row
        try:
            cursor = await conn.execute(
                """
                SELECT id, text, date, event_ts_hint
                FROM vk_inbox
                WHERE status IN ('pending', 'locked', 'skipped')
                """
            )
            rows = await cursor.fetchall()
            await cursor.close()

            for row in rows:
                inbox_id = row["id"]
                text = row["text"] or ""
                publish_ts = row["date"]
                try:
                    hint = extract_event_ts_hint(
                        text, publish_ts=publish_ts, allow_past=True
                    )
                except Exception:  # pragma: no cover - defensive
                    logging.exception(
                        "vk_review refresh_hint_failed id=%s", inbox_id
                    )
                    hint = None
                if hint != row["event_ts_hint"]:
                    updates.append((hint, inbox_id))

            for hint, inbox_id in updates:
                await conn.execute(
                    "UPDATE vk_inbox SET event_ts_hint=? WHERE id=?",
                    (hint, inbox_id),
                )

            if updates:
                await conn.commit()
        finally:
            conn.row_factory = original_row_factory

    if updates:
        logging.info(
            "vk_review refresh_vk_event_ts_hints updated=%s", len(updates)
        )
    return len(updates)


_FAR_BUCKET_HISTORY: dict[int, deque[str]] = {}


@dataclass
class InboxPost:
    id: int
    group_id: int
    post_id: int
    date: int
    text: str
    matched_kw: Optional[str]
    has_date: int
    status: str
    review_batch: Optional[str]
    imported_event_id: Optional[int]
    event_ts_hint: Optional[int]


def _hours_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning("vk_review invalid env %s=%s, using default %s", name, value, default)
        return default


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logging.warning("vk_review invalid env %s=%s, using default %s", name, value, default)
        return default


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("vk_review invalid env %s=%s, using default %s", name, value, default)
        return default


def _get_far_history(operator_id: int, limit: int) -> Optional[deque[str]]:
    if limit <= 0:
        _FAR_BUCKET_HISTORY.pop(operator_id, None)
        return None
    history = _FAR_BUCKET_HISTORY.get(operator_id)
    if history is None or history.maxlen != limit:
        history = deque(maxlen=limit)
        _FAR_BUCKET_HISTORY[operator_id] = history
    return history


async def pick_next(db: Database, operator_id: int, batch_id: str) -> Optional[InboxPost]:
    """Select the next inbox item and lock it for the operator.

    Rows in ``pending`` state are preferred.  When none are available, all
    rows in ``skipped`` state are moved back to ``pending`` and the selection is
    repeated.  Items are ordered by ``event_ts_hint`` ascending and, within the
    same hint, by ``date`` and ``id`` descending.  The selected row is
    atomically updated to ``locked`` state with ``locked_by`` and ``locked_at``
    set and ``review_batch`` recorded so later imports can accumulate months
    for this batch.

    ``None`` is returned when the queue is empty.
    """

    async with db.raw_conn() as conn:
        await _unlock_stale(conn)

        reject_window_hours = _hours_from_env("VK_REVIEW_REJECT_H", 2)
        urgent_window_hours = _hours_from_env("VK_REVIEW_URGENT_MAX_H", 48)
        urgent_window_hours = max(urgent_window_hours, reject_window_hours)

        now_ts = int(_time.time())
        reject_cutoff = now_ts + int(reject_window_hours * 3600)

        while True:
            # Continue reviewing rows that remain locked for this operator.
            cur = await conn.execute(
                """
                SELECT id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch, imported_event_id, event_ts_hint
                FROM vk_inbox
                WHERE status='locked' AND locked_by=?
                ORDER BY locked_at ASC, id ASC
                LIMIT 1
                """,
                (operator_id,),
            )
            row = await cur.fetchone()
            if not row:
                break

            inbox_id = row[0]
            text = row[4]
            matched_kw = row[5]
            has_date = row[6]
            skip_hint_recalc = matched_kw == OCR_PENDING_SENTINEL and has_date == 0
            publish_ts = row[3]
            ts_hint = (
                None
                if skip_hint_recalc
                else extract_event_ts_hint(text, publish_ts=publish_ts)
            )
            if not skip_hint_recalc and (ts_hint is None or ts_hint < reject_cutoff):
                await conn.execute(
                    "UPDATE vk_inbox SET status='rejected', locked_by=NULL, locked_at=NULL, review_batch=NULL WHERE id=?",
                    (inbox_id,),
                )
                await conn.commit()
                logging.info(
                    "vk_review reject_locked_due_to_hint id=%s operator=%s hint=%s cutoff=%s",
                    inbox_id,
                    operator_id,
                    ts_hint,
                    reject_cutoff,
                )
                now_ts = int(_time.time())
                reject_cutoff = now_ts + int(reject_window_hours * 3600)
                continue

            if not skip_hint_recalc and ts_hint is not None:
                await conn.execute(
                    "UPDATE vk_inbox SET event_ts_hint=?, review_batch=?, locked_at=CURRENT_TIMESTAMP WHERE id=?",
                    (ts_hint, batch_id, inbox_id),
                )
            else:
                await conn.execute(
                    "UPDATE vk_inbox SET review_batch=?, locked_at=CURRENT_TIMESTAMP WHERE id=?",
                    (batch_id, inbox_id),
                )
            await conn.commit()
            row = list(row)
            row[8] = batch_id
            row[10] = ts_hint
            post = InboxPost(*row)
            logging.info(
                "vk_review resume_locked id=%s operator=%s batch=%s",
                post.id,
                operator_id,
                batch_id,
            )
            return post

        selected_row = None
        final_bucket_name: Optional[str] = None
        final_weight_config: dict[str, float] = {}
        far_gap_k = max(_int_from_env("VK_REVIEW_FAR_GAP_K", 5), 0)
        history = _get_far_history(operator_id, far_gap_k)
        while True:
            bucket_name_for_history: Optional[str] = None
            weight_config_for_log: dict[str, float] = {}
            now_ts = int(_time.time())
            reject_cutoff = now_ts + int(reject_window_hours * 3600)
            urgent_cutoff = now_ts + int(urgent_window_hours * 3600)
            await conn.execute(
                "UPDATE vk_inbox SET status='rejected', locked_by=NULL, locked_at=NULL WHERE status IN ('pending','skipped') AND event_ts_hint IS NOT NULL AND event_ts_hint < ?",
                (reject_cutoff,),
            )
            cur = await conn.execute(
                "SELECT 1 FROM vk_inbox WHERE status='pending' AND (event_ts_hint IS NULL OR event_ts_hint >= ?) LIMIT 1",
                (reject_cutoff,),
            )
            has_pending = await cur.fetchone() is not None
            if not has_pending:
                await conn.execute(
                    "UPDATE vk_inbox SET status='pending' WHERE status='skipped' AND (event_ts_hint IS NULL OR event_ts_hint >= ?)",
                    (reject_cutoff,),
                )

            cursor = await conn.execute(
                """
                WITH next AS (
                    SELECT id FROM vk_inbox
                    WHERE status='pending'
                      AND event_ts_hint IS NOT NULL
                      AND event_ts_hint >= ?
                      AND event_ts_hint < ?
                    ORDER BY event_ts_hint ASC, date DESC, id DESC
                    LIMIT 1
                )
                UPDATE vk_inbox
                SET status='locked', locked_by=?, locked_at=CURRENT_TIMESTAMP, review_batch=?
                WHERE id = (SELECT id FROM next)
                RETURNING id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch, imported_event_id, event_ts_hint
                """,
                (reject_cutoff, urgent_cutoff, operator_id, batch_id),
            )
            row = await cursor.fetchone()
            if row:
                bucket_name_for_history = "URGENT"
            if not row:
                soon_max_days = max(_float_from_env("VK_REVIEW_SOON_MAX_D", 14), 0.0)
                long_max_days = max(
                    _float_from_env("VK_REVIEW_LONG_MAX_D", 30),
                    soon_max_days,
                )
                soon_cutoff = max(urgent_cutoff, now_ts + int(soon_max_days * 86400))
                long_cutoff = max(soon_cutoff, now_ts + int(long_max_days * 86400))

                bucket_specs = [
                    (
                        "SOON",
                        "status='pending' AND event_ts_hint IS NOT NULL AND event_ts_hint >= ? AND event_ts_hint < ?",
                        (urgent_cutoff, soon_cutoff),
                        max(_float_from_env("VK_REVIEW_W_SOON", 3.0), 0.0),
                    ),
                    (
                        "LONG",
                        "status='pending' AND event_ts_hint IS NOT NULL AND event_ts_hint >= ? AND event_ts_hint < ?",
                        (soon_cutoff, long_cutoff),
                        max(_float_from_env("VK_REVIEW_W_LONG", 2.0), 0.0),
                    ),
                    (
                        "FAR",
                        "status='pending' AND (event_ts_hint IS NULL OR event_ts_hint >= ?)",
                        (long_cutoff,),
                        max(_float_from_env("VK_REVIEW_W_FAR", 6.0), 0.0),
                    ),
                ]

                bucket_counts: dict[str, int] = {}
                bucket_specs_by_name: dict[str, tuple[str, tuple[Any, ...]]] = {}
                weighted_total = 0.0
                weight_config_for_log = {
                    name: weight for name, _, _, weight in bucket_specs
                }
                for name, where_clause, params, weight in bucket_specs:
                    count_cursor = await conn.execute(
                        f"SELECT COUNT(1) FROM vk_inbox WHERE {where_clause}",
                        params,
                    )
                    count_row = await count_cursor.fetchone()
                    count = int(count_row[0]) if count_row else 0
                    bucket_counts[name] = count
                    bucket_specs_by_name[name] = (where_clause, params)
                    weighted_total += weight * count

                chosen_bucket = None
                if (
                    history is not None
                    and history.maxlen
                    and len(history) == history.maxlen
                    and all(bucket != "FAR" for bucket in history)
                    and bucket_counts.get("FAR", 0) > 0
                    and "FAR" in bucket_specs_by_name
                ):
                    where_clause, params = bucket_specs_by_name["FAR"]
                    chosen_bucket = ("FAR", where_clause, params)
                    logging.info(
                        "vk_review far_gap_override operator=%s history=%s counts=%s",
                        operator_id,
                        list(history),
                        bucket_counts,
                    )
                elif weighted_total > 0:
                    ticket = random.random() * weighted_total
                    for name, where_clause, params, weight in bucket_specs:
                        count = bucket_counts.get(name, 0)
                        bucket_weight = weight * count
                        if bucket_weight <= 0:
                            continue
                        if ticket < bucket_weight:
                            chosen_bucket = (name, where_clause, params)
                            break
                        ticket -= bucket_weight

                if chosen_bucket:
                    name, where_clause, params = chosen_bucket
                    logging.info(
                        "vk_review bucket_pick name=%s counts=%s", name, bucket_counts
                    )
                    penalty_cursor = await conn.execute(
                        f"SELECT group_id, COUNT(*) FROM vk_inbox WHERE {where_clause} GROUP BY group_id",
                        params,
                    )
                    penalty_rows = await penalty_cursor.fetchall()
                    await penalty_cursor.close()
                    penalty_params: list[float | int] = []
                    penalty_cte = ", group_penalties(group_id, penalty) AS (SELECT NULL, 1.0 LIMIT 0)"
                    if penalty_rows:
                        values_clause = ", ".join(["(?, ?)"] * len(penalty_rows))
                        penalty_cte = (
                            f", group_penalties(group_id, penalty) AS (VALUES {values_clause})"
                        )
                        for group_id, cnt in penalty_rows:
                            penalty_params.extend([group_id, math.sqrt(cnt)])

                    bucket_query = f"""
                        WITH candidates AS (
                            SELECT id, group_id, event_ts_hint, date
                            FROM vk_inbox
                            WHERE {where_clause}
                        ){penalty_cte},
                        ranked AS (
                            SELECT c.id
                            FROM candidates c
                            LEFT JOIN group_penalties gp ON c.group_id = gp.group_id
                            ORDER BY c.event_ts_hint ASC,
                                     (ABS(RANDOM()) / 9223372036854775808.0) * 0.001 *
                                         COALESCE(gp.penalty, 1.0) ASC,
                                     c.date DESC,
                                     c.id DESC
                            LIMIT 1
                        )
                        UPDATE vk_inbox
                        SET status='locked', locked_by=?, locked_at=CURRENT_TIMESTAMP, review_batch=?
                        WHERE id = (SELECT id FROM ranked)
                        RETURNING id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch, imported_event_id, event_ts_hint
                    """
                    bucket_cursor = await conn.execute(
                        bucket_query,
                        (*params, *penalty_params, operator_id, batch_id),
                    )
                    row = await bucket_cursor.fetchone()
                    if row:
                        bucket_name_for_history = name

                if not row:
                    cursor = await conn.execute(
                        """
                        WITH next AS (
                            SELECT id FROM vk_inbox
                            WHERE status='pending' AND (event_ts_hint IS NULL OR event_ts_hint >= ?)
                            ORDER BY CASE WHEN event_ts_hint IS NULL THEN 1 ELSE 0 END,
                                     event_ts_hint ASC, date DESC, id DESC
                            LIMIT 1
                        )
                        UPDATE vk_inbox
                        SET status='locked', locked_by=?, locked_at=CURRENT_TIMESTAMP, review_batch=?
                        WHERE id = (SELECT id FROM next)
                        RETURNING id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch, imported_event_id, event_ts_hint
                        """,
                        (reject_cutoff, operator_id, batch_id),
                    )
                    row = await cursor.fetchone()
                    if not row:
                        await conn.commit()
                        return None
                    bucket_name_for_history = "FALLBACK"

            inbox_id = row[0]
            text = row[4]
            matched_kw = row[5]
            has_date = row[6]
            skip_hint_recalc = matched_kw == OCR_PENDING_SENTINEL and has_date == 0
            publish_ts = row[3]
            ts_hint = (
                None
                if skip_hint_recalc
                else extract_event_ts_hint(text, publish_ts=publish_ts)
            )
            if not skip_hint_recalc and (ts_hint is None or ts_hint < reject_cutoff):
                await conn.execute(
                    "UPDATE vk_inbox SET status='rejected', locked_by=NULL, locked_at=NULL, review_batch=NULL WHERE id=?",
                    (inbox_id,),
                )
                await conn.commit()
                logging.info(
                    "vk_review reject_due_to_hint id=%s operator=%s batch=%s", inbox_id, operator_id, batch_id
                )
                continue

            if not skip_hint_recalc and ts_hint is not None:
                await conn.execute(
                    "UPDATE vk_inbox SET event_ts_hint=? WHERE id=?",
                    (ts_hint, inbox_id),
                )
            await conn.commit()
            selected_row = list(row)
            if not skip_hint_recalc:
                selected_row[10] = ts_hint
            final_bucket_name = bucket_name_for_history
            final_weight_config = weight_config_for_log
            if final_bucket_name and history is not None:
                history.append(final_bucket_name)
            break
    post = InboxPost(*selected_row)
    logging.info(
        "vk_review pick_next id=%s group=%s post=%s kw=%s has_date=%s bucket=%s weights=%s",
        post.id,
        post.group_id,
        post.post_id,
        post.matched_kw,
        post.has_date,
        final_bucket_name,
        final_weight_config,
    )
    return post


async def mark_skipped(db: Database, inbox_id: int) -> None:
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_inbox SET status='skipped', locked_by=NULL, locked_at=NULL WHERE id=?",
            (inbox_id,),
        )
        await conn.commit()


async def mark_rejected(db: Database, inbox_id: int) -> None:
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_inbox SET status='rejected', locked_by=NULL, locked_at=NULL WHERE id=?",
            (inbox_id,),
        )
        await conn.commit()


async def mark_imported(
    db: Database,
    inbox_id: int,
    batch_id: str,
    operator_id: int,
    event_id: int | None,
    event_date: str | None,
) -> None:
    """Mark inbox row as imported and accumulate affected month.

    ``event_date`` should be in ISO ``YYYY-MM-DD`` format.  The month part is
    stored in ``vk_review_batch.months_csv`` as a comma separated set.  The
    row is unlocked and linked with ``event_id`` (which may be ``None`` when
    only a festival record is created).
    """

    month = (event_date or "")[:7]
    async with db.raw_conn() as conn:
        if not batch_id:
            cur = await conn.execute(
                "SELECT review_batch FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
            if row and row[0]:
                batch_id = row[0]

        await conn.execute(
            """
            UPDATE vk_inbox
            SET status='imported', locked_by=NULL, locked_at=NULL,
                imported_event_id=?, review_batch=?
            WHERE id=?
            """,
            (event_id, batch_id, inbox_id),
        )
        if batch_id:
            await conn.execute(
                """
                INSERT OR IGNORE INTO vk_review_batch(batch_id, operator_id, months_csv)
                VALUES(?,?,?)
                """,
                (batch_id, operator_id, ""),
            )
            cur = await conn.execute(
                "SELECT months_csv FROM vk_review_batch WHERE batch_id=?",
                (batch_id,),
            )
            row = await cur.fetchone()
            months: set[str] = set()
            if row and row[0]:
                months = set(filter(None, row[0].split(',')))
            if month:
                months.add(month)
            months_csv = ",".join(sorted(months))
            await conn.execute(
                "UPDATE vk_review_batch SET months_csv=?, finished_at=NULL WHERE batch_id=?",
                (months_csv, batch_id),
            )
        else:
            logging.warning(
                "vk_review mark_imported missing_batch",
                extra={
                    "inbox_id": inbox_id,
                    "event_id": event_id,
                    "event_date": event_date,
                },
            )
        await conn.commit()
    logging.info(
        "vk_review mark_imported inbox_id=%s event_id=%s month=%s",
        inbox_id,
        event_id,
        month,
    )


async def save_repost_url(db: Database, event_id: int, url: str) -> None:
    """Persist ``vk_repost_url`` for the event."""

    async with db.raw_conn() as conn:
        async def _update() -> None:
            await conn.execute(
                "UPDATE event SET vk_repost_url=? WHERE id=?",
                (url, event_id),
            )
            await conn.commit()

        await _retry_locked_write(
            conn,
            _update,
            description=f"save_repost_url event_id={event_id}",
        )


async def finish_batch(
    db: Database,
    batch_id: str,
    rebuild_cb: Callable[[Database, str], Awaitable[Any]],
) -> list[str]:
    """Finish review batch and rebuild affected months sequentially.

    ``rebuild_cb`` is awaited for every month individually to guarantee
    sequential rebuilds.  The function clears ``months_csv`` and sets
    ``finished_at`` timestamp.  Returns the list of months that were rebuilt.
    """

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT months_csv FROM vk_review_batch WHERE batch_id=?", (batch_id,)
        )
        row = await cur.fetchone()
        months = [m for m in (row[0].split(',') if row and row[0] else []) if m]
    for month in months:
        start = _time.perf_counter() if "_time" in globals() else None
        await rebuild_cb(db, month)
        if start is not None:
            took = int((_time.perf_counter() - start) * 1000)
            logging.info("vk_review rebuild month=%s took_ms=%d", month, took)
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_review_batch SET months_csv='', finished_at=CURRENT_TIMESTAMP WHERE batch_id=?",
            (batch_id,),
        )
        await conn.commit()
    return months
