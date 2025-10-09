import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from db import Database


@pytest.mark.asyncio
async def test_add_events_from_text_vk_failure(tmp_path: Path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    future = (date.today() + timedelta(days=30)).isoformat()

    async def fake_parse(text, *args, **kwargs):
        return main.ParsedEvents(
            [],
            festival={
                "name": "Fest",
                "start_date": future,
                "location_name": "Hall",
                "city": "Town",
            },
        )

    async def fake_upload(images):
        return [], ""

    async def fake_sync_page(db_obj, name, **kwargs):
        pass

    async def fake_sync_vk(db_obj, name, bot_obj, strict=False):
        raise RuntimeError("vk fail")

    notified: dict[str, str] = {}

    async def fake_notify(db_obj, bot_obj, text):
        notified["text"] = text

    async def fake_rebuild(*args, **kwargs):
        return "skipped", ""

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "upload_images", fake_upload)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_vk)
    monkeypatch.setattr(main, "notify_superadmin", fake_notify)
    monkeypatch.setattr(main, "rebuild_festivals_index_if_needed", fake_rebuild)

    with caplog.at_level(logging.ERROR):
        res = await main.add_events_from_text(db, "t", None, None, None, bot=object())
        await asyncio.sleep(0)

    fest = res[0][0]
    assert isinstance(fest, main.Festival)
    assert any("festival VK sync failed for Fest" in r.message for r in caplog.records)
    assert notified["text"].startswith("festival VK sync failed for Fest")

