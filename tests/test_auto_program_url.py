import pytest
from pathlib import Path
from sqlalchemy import select

import main
from db import Database
from models import Festival


@pytest.mark.asyncio
async def test_add_events_from_text_autofills_program_url(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text, *args, **kwargs):
        return main.ParsedEvents([], festival={"name": "Fest"})

    async def fake_upload(images):
        return [], ""

    async def fake_sync_page(db, name):
        return None

    async def fake_sync_vk(db, name, bot, strict=False):
        return None

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "upload_images", fake_upload)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_vk)
    async def fake_extract(url, *, event_id=None):
        return None

    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)
    async def fake_rebuild(*a, **k):
        return "built", ""

    monkeypatch.setattr(main, "rebuild_festivals_index_if_needed", fake_rebuild)

    html = '<a href="https://telegra.ph/prog">prog</a>'
    await main.add_events_from_text(db, "t", None, html, None)

    async with db.get_session() as session:
        fest = await session.scalar(select(Festival).where(Festival.name == "Fest"))
        assert fest is not None
        assert fest.program_url == "https://telegra.ph/prog"

    html2 = '<a href="https://example.com/program">prog</a>'
    await main.add_events_from_text(db, "t", None, html2, None)

    async with db.get_session() as session:
        fest = await session.scalar(select(Festival).where(Festival.name == "Fest"))
        assert fest.program_url == "https://example.com/program"

