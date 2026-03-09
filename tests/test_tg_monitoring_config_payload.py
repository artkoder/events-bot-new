from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from db import Database
from models import TelegramScannedMessage, TelegramSource
from source_parsing.telegram.service import _build_config_payload


@pytest.mark.asyncio
async def test_tg_monitoring_config_payload_includes_only_recent_eventful_message_ids(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("TG_MONITORING_DAYS_BACK", "3")
    monkeypatch.setenv("TG_MONITORING_RECENT_RESCAN_LIMIT", "10")

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        now = datetime.now(timezone.utc)
        async with db.get_session() as session:
            src = TelegramSource(
                username="recent_eventful_chan",
                enabled=True,
                last_scanned_message_id=200,
            )
            session.add(src)
            await session.flush()
            session.add_all(
                [
                    TelegramScannedMessage(
                        source_id=int(src.id),
                        message_id=101,
                        message_date=now - timedelta(days=1),
                        status="done",
                        events_extracted=1,
                        events_imported=1,
                    ),
                    TelegramScannedMessage(
                        source_id=int(src.id),
                        message_id=102,
                        message_date=now - timedelta(days=2),
                        status="done",
                        events_extracted=0,
                        events_imported=1,
                    ),
                    TelegramScannedMessage(
                        source_id=int(src.id),
                        message_id=103,
                        message_date=now - timedelta(days=1),
                        status="done",
                        events_extracted=0,
                        events_imported=0,
                    ),
                    TelegramScannedMessage(
                        source_id=int(src.id),
                        message_id=104,
                        message_date=now - timedelta(days=5),
                        status="done",
                        events_extracted=1,
                        events_imported=1,
                    ),
                ]
            )
            await session.commit()

        payload = await _build_config_payload(db, run_id="r-test")
        source_payload = next(
            item for item in (payload.get("sources") or []) if item.get("username") == "recent_eventful_chan"
        )
        assert sorted(source_payload.get("recent_event_message_ids") or []) == [101, 102]
    finally:
        await db.close()
