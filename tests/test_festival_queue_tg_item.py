from pathlib import Path

import pytest

import festival_queue
import main
from db import Database
from models import Festival, FestivalQueueItem, TelegramSource, TelegramSourceForceMessage
from sqlmodel import select


@pytest.mark.asyncio
async def test_process_tg_item_ensures_festival_stub_on_empty_monitor(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    item = FestivalQueueItem(
        source_kind="tg",
        source_url="https://t.me/terkatalk/4483",
        source_text="ПРИРОДОВИДЕНИЕ 7/02 — 26/02",
        source_chat_username="terkatalk",
        source_chat_id=1812770807,
        source_message_id=4483,
        festival_context="festival_post",
        festival_name="ПриродоВидение",
    )
    async with db.get_session() as session:
        session.add(item)
        await session.commit()
        await session.refresh(item)

    calls = {"monitor": 0, "ensure": 0}

    async def fake_monitor(*args, **kwargs):
        calls["monitor"] += 1
        return None

    async def fake_ensure_festival(
        db_obj,
        name,
        *,
        full_name=None,
        tg_url=None,
        source_text=None,
        source_post_url=None,
        source_chat_id=None,
        source_message_id=None,
        **kwargs,
    ):
        calls["ensure"] += 1
        assert db_obj is db
        assert name == "ПриродоВидение"
        assert tg_url is None
        assert source_post_url == "https://t.me/terkatalk/4483"
        assert source_chat_id == 1812770807
        assert source_message_id == 4483
        fest = Festival(name="ПриродоВидение", full_name=full_name, telegraph_path="prirodovidenie")
        return fest, True, False

    import source_parsing.telegram.service as tg_service

    monkeypatch.setattr(tg_service, "run_telegram_monitor", fake_monitor)
    monkeypatch.setattr(main, "ensure_festival", fake_ensure_festival)

    result = await festival_queue._process_tg_item(db, item, bot=None, chat_id=None)
    assert result["festival_name"] == "ПриродоВидение"
    assert calls == {"monitor": 1, "ensure": 1}

    async with db.get_session() as session:
        source = (
            await session.execute(
                select(TelegramSource).where(TelegramSource.username == "terkatalk")
            )
        ).scalar_one_or_none()
        assert source is not None
        assert source.username == "terkatalk"
        forced = await session.get(TelegramSourceForceMessage, (source.id, 4483))
        assert forced is not None


@pytest.mark.asyncio
async def test_process_tg_item_falls_back_when_monitor_fails(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    item = FestivalQueueItem(
        source_kind="tg",
        source_url="https://t.me/terkatalk/4483",
        source_text="ПРИРОДОВИДЕНИЕ 7/02 — 26/02",
        source_chat_username="terkatalk",
        source_chat_id=1812770807,
        source_message_id=4483,
        festival_context="festival_post",
        festival_name="ПриродоВидение",
    )

    calls = {"monitor": 0, "ensure": 0}

    async def fake_monitor(*args, **kwargs):
        calls["monitor"] += 1
        raise RuntimeError("Kaggle kernel failed (failed)")

    async def fake_ensure_festival(db_obj, name, **kwargs):
        calls["ensure"] += 1
        return Festival(name=name, telegraph_path="prirodovidenie"), True, False

    import source_parsing.telegram.service as tg_service

    monkeypatch.setattr(tg_service, "run_telegram_monitor", fake_monitor)
    monkeypatch.setattr(main, "ensure_festival", fake_ensure_festival)

    result = await festival_queue._process_tg_item(db, item, bot=None, chat_id=None)
    assert calls == {"monitor": 1, "ensure": 1}
    assert result["festival_name"] == "ПриродоВидение"
    assert result["mode"] == "fallback_ensure_festival"
    assert "Kaggle kernel failed" in result["monitor_error"]
