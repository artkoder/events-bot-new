from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from db import Database


@dataclass
class InboxPost:
    id: int
    group_id: int
    post_id: int
    date: int
    text: str
    status: str
    review_batch: Optional[str]


async def pick_next(db: Database, operator_id: int, batch_id: str) -> Optional[InboxPost]:
    """Select the next inbox item and lock it for the operator.

    Rows with status ``pending`` or ``skipped`` are considered, ordered by
    ``date`` and ``id`` descending.  The selected row is atomically updated to
    ``locked`` state with ``locked_by`` and ``locked_at`` set.  ``review_batch``
    is also recorded so later imports can accumulate months for this batch.

    ``None`` is returned when the queue is empty.
    """

    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            """
            WITH next AS (
                SELECT id FROM vk_inbox
                WHERE status IN ('pending','skipped')
                ORDER BY date DESC, id DESC
                LIMIT 1
            )
            UPDATE vk_inbox
            SET status='locked', locked_by=?, locked_at=CURRENT_TIMESTAMP, review_batch=?
            WHERE id = (SELECT id FROM next)
            RETURNING id, group_id, post_id, date, text, status, review_batch
            """,
            (operator_id, batch_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        await conn.commit()
    return InboxPost(*row)


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


async def finish_batch(
    db: Database, batch_id: str, rebuild_cb: Any
) -> list[str]:
    """Finish review batch and rebuild affected months.

    ``rebuild_cb`` is a callable accepting ``(db, months)`` and returning a
    report string.  The function clears ``months_csv`` and sets ``finished_at``
    timestamp.  Returns the list of months that were rebuilt.
    """

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT months_csv FROM vk_review_batch WHERE batch_id=?", (batch_id,)
        )
        row = await cur.fetchone()
        months = [m for m in (row[0].split(',') if row and row[0] else []) if m]
    if months:
        await rebuild_cb(db, months)
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_review_batch SET months_csv='', finished_at=CURRENT_TIMESTAMP WHERE batch_id=?",
            (batch_id,),
        )
        await conn.commit()
    return months
