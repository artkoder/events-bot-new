from __future__ import annotations

import logging
import os
from typing import Any

from db import Database

logger = logging.getLogger(__name__)


def parse_chat_id(raw: Any, *, label: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        logger.warning("invalid %s=%r", label, raw)
        return None


async def resolve_superadmin_chat_id(
    db: Database | None,
    *,
    env_var: str = "ADMIN_CHAT_ID",
) -> int | None:
    """Resolve the chat to notify for scheduled/background admin reports.

    Primary source is the registered superadmin in SQLite. The env fallback is kept
    only for early/bootstrap environments where the DB does not yet contain a
    superadmin row.
    """

    if db is not None and hasattr(db, "raw_conn"):
        try:
            async with db.raw_conn() as conn:
                cur = await conn.execute(
                    """
                    SELECT user_id
                    FROM "user"
                    WHERE is_superadmin = 1
                      AND COALESCE(blocked, 0) = 0
                    ORDER BY user_id ASC
                    LIMIT 1
                    """
                )
                row = await cur.fetchone()
                if row and row[0] is not None:
                    return int(row[0])
        except Exception:
            logger.debug("admin_chat: failed to resolve superadmin from database", exc_info=True)

    return parse_chat_id(os.getenv(env_var), label=env_var)
