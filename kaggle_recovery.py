from __future__ import annotations

import logging

from db import Database
from video_announce.poller import resume_rendering_sessions
from preview_3d.handlers import resume_preview3d_jobs
from source_parsing.handlers import resume_source_parsing_jobs
from source_parsing.telegram.service import resume_telegram_monitor_jobs

logger = logging.getLogger(__name__)


async def kaggle_recovery_scheduler(
    db: Database,
    bot,
    *,
    run_id: str | None = None,
) -> None:
    try:
        recovered_v = await resume_rendering_sessions(db, bot)
        recovered_3d = await resume_preview3d_jobs(db, bot)
        recovered_parse = await resume_source_parsing_jobs(db, bot)
        recovered_tg = await resume_telegram_monitor_jobs(db, bot)
        if recovered_v or recovered_3d or recovered_parse or recovered_tg:
            logger.info(
                "kaggle_recovery run_id=%s video=%s 3di=%s parse=%s tg_monitor=%s",
                run_id,
                recovered_v,
                recovered_3d,
                recovered_parse,
                recovered_tg,
            )
    except Exception:
        logger.exception("kaggle_recovery failed run_id=%s", run_id)
