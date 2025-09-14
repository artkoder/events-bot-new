from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Awaitable, Callable

import logging
import time as _time

from db import Database


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

    cutoff = int(_time.time()) + 2 * 3600
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_inbox SET status='rejected', locked_by=NULL, locked_at=NULL WHERE status IN ('pending','skipped') AND event_ts_hint IS NOT NULL AND event_ts_hint < ?",
            (cutoff,),
        )
        cur = await conn.execute(
            "SELECT 1 FROM vk_inbox WHERE status='pending' AND (event_ts_hint IS NULL OR event_ts_hint >= ?) LIMIT 1",
            (cutoff,),
        )
        has_pending = await cur.fetchone() is not None
        if not has_pending:
            await conn.execute(
                "UPDATE vk_inbox SET status='pending' WHERE status='skipped' AND (event_ts_hint IS NULL OR event_ts_hint >= ?)",
                (cutoff,),
            )

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
            RETURNING id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch
            """,
            (cutoff, operator_id, batch_id),
        )
        row = await cursor.fetchone()
        await conn.commit()
        if not row:
            return None
    post = InboxPost(*row)
    logging.info(
        "vk_review pick_next id=%s group=%s post=%s kw=%s has_date=%s",
        post.id,
        post.group_id,
        post.post_id,
        post.matched_kw,
        post.has_date,
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
    db: Database, inbox_id: int, batch_id: str, event_id: int, event_date: str
) -> None:
    """Mark inbox row as imported and accumulate affected month.

    ``event_date`` should be in ISO ``YYYY-MM-DD`` format.  The month part is
    stored in ``vk_review_batch.months_csv`` as a comma separated set.  The
    row is unlocked and linked with ``event_id``.
    """

    month = event_date[:7]
    async with db.raw_conn() as conn:
        await conn.execute(
            """
            UPDATE vk_inbox
            SET status='imported', locked_by=NULL, locked_at=NULL,
                imported_event_id=?, review_batch=?
            WHERE id=?
            """,
            (event_id, batch_id, inbox_id),
        )
        cur = await conn.execute(
            "SELECT months_csv FROM vk_review_batch WHERE batch_id=?", (batch_id,)
        )
        row = await cur.fetchone()
        months: set[str] = set()
        if row and row[0]:
            months = set(filter(None, row[0].split(',')))
        months.add(month)
        months_csv = ",".join(sorted(months))
        await conn.execute(
            "UPDATE vk_review_batch SET months_csv=? WHERE batch_id=?",
            (months_csv, batch_id),
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
        await conn.execute(
            "UPDATE event SET vk_repost_url=? WHERE id=?",
            (url, event_id),
        )
        await conn.commit()


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
