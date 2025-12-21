from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from db import Database
from models import VideoAnnounceSession, VideoAnnounceSessionStatus

logger = logging.getLogger(__name__)


async def reset_stuck_sessions(db: Database, *, max_age_minutes: int = 30) -> int:
    """Move long-running RENDERING sessions into FAILED state."""

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    updated = 0
    async with db.get_session() as session:
        res = await session.execute(
            select(VideoAnnounceSession).where(
                VideoAnnounceSession.status == VideoAnnounceSessionStatus.RENDERING,
                VideoAnnounceSession.started_at < cutoff,
            )
        )
        for sess in res.scalars():
            sess.status = VideoAnnounceSessionStatus.FAILED
            sess.finished_at = datetime.now(timezone.utc)
            sess.error = "stuck rendering watchdog"
            updated += 1
            logger.warning("video_announce reset stuck session_id=%s", sess.id)
        if updated:
            await session.commit()
    return updated
