from __future__ import annotations

import logging

from db import Database
from video_announce.poller import resume_rendering_sessions
from preview_3d.handlers import resume_preview3d_jobs
from source_parsing.handlers import resume_source_parsing_jobs

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
        if recovered_v or recovered_3d or recovered_parse:
            logger.info(
                "kaggle_recovery run_id=%s video=%s 3di=%s parse=%s",
                run_id,
                recovered_v,
                recovered_3d,
                recovered_parse,
            )
    except Exception:
        logger.exception("kaggle_recovery failed run_id=%s", run_id)
