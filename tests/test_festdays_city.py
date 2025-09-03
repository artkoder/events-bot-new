import os
import sys
from pathlib import Path
from datetime import date, timedelta

import os
import sys
from pathlib import Path
from datetime import date, timedelta

import pytest
from aiogram import types, Bot
from sqlmodel import select

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import main
from main import (
    Database,
    Festival,
    Event,
    ensure_festival,
    upsert_event,
    schedule_event_update_tasks,
    parse_city_from_fest_name,
    parse_iso_date,
    process_request,
)


class DummyBot(Bot):
    async def request(self, method, data=None, files=None):  # pragma: no cover - network stub
        return None


@pytest.mark.asyncio
async def test_ensure_festival_updates_city_when_changed(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # initial festival
    fest, created, updated = await ensure_festival(db, "Fest", city="Черняховск")
    assert created and updated
    # update with new city
    fest2, created2, updated2 = await ensure_festival(db, "Fest", city="Зеленоградск")
    assert not created2
    assert updated2
    assert fest2.city == "Зеленоградск"


@pytest.mark.asyncio
async def test_festdays_uses_correct_city(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    start_day = date(2024, 9, 6)
    async with db.get_session() as session:
        fest = Festival(
            name="День города Зеленоградск",
            start_date=start_day.isoformat(),
            end_date=(start_day + timedelta(days=1)).isoformat(),
        )
        session.add(fest)
        await session.commit()
        fid = fest.id

    # stub sync and notifications
    async def fake_sync_month_page(db_obj, month):
        pass

    async def fake_sync_weekend_page(db_obj, start):
        pass

    async def fake_sync_festival_page(db_obj, name, **kwargs):
        pass

    async def fake_sync_vk(db_obj, name, bot_obj, strict=False):
        pass

    async def fake_notify(db_obj, bot_obj, user, event, added):
        pass

    monkeypatch.setattr(main, "sync_month_page", fake_sync_month_page)
    monkeypatch.setattr(main, "sync_weekend_page", fake_sync_weekend_page)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_festival_page)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_vk)
    monkeypatch.setattr(main, "notify_event_added", fake_notify)
    async def fake_show_menu(*a, **k):
        return None
    monkeypatch.setattr(main, "show_festival_edit_menu", fake_show_menu)
    assert main.show_festival_edit_menu is fake_show_menu
    async def fake_send_message(*a, **k):
        return None
    monkeypatch.setattr(bot, "send_message", fake_send_message)

    async def fake_show_menu(*a, **k):
        return None

    monkeypatch.setattr(main, "show_festival_edit_menu", fake_show_menu)

    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "data": f"festdays:{fid}",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "chat_instance": "1",
            "message": {
                "message_id": 1,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
                "text": "stub",
            },
        }
    ).as_(bot)

    async def dummy_answer(text=None, **kwargs):
        return None

    object.__setattr__(cb, "answer", dummy_answer)
    object.__setattr__(cb.message, "answer", dummy_answer)

    await process_request(cb, db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        assert len(events) == 2
        assert all(e.city == "Зеленоградск" for e in events)
        assert all(e.source_text.startswith("День города Зеленоградск —") for e in events)


@pytest.mark.asyncio
async def test_create_source_page_no_dedupe_on_empty_text(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # insert existing event with empty source_text and telegraph_path
    async with db.get_session() as session:
        ev = Event(
            title="T",
            description="",
            festival="",
            date="2024-01-01",
            time="",
            location_name="",
            source_text="",
            telegraph_path="exist",
            telegraph_url="u",
        )
        session.add(ev)
        await session.commit()

    counter = 0

    async def fake_create_page(tg, *args, **kwargs):
        nonlocal counter
        counter += 1
        return {"path": f"p{counter}", "url": f"http://t.me/p{counter}"}

    async def fake_build(*args, **kwargs):
        return "", "", 0

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "build_source_page_content", fake_build)
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None, domain=None: object())

    res1 = await main.create_source_page("T1", "", None, db=db)
    res2 = await main.create_source_page("T2", "", None, db=db)

    assert res1[1] != "exist"
    assert res2[1] != res1[1]


@pytest.mark.asyncio
async def test_festdays_two_cities_no_mixup(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async with db.get_session() as session:
        fest1 = Festival(
            name="День города Зеленоградск",
            start_date="2024-09-06",
            end_date="2024-09-07",
            city="Зеленоградск",
            location_name="Зеленоградск",
        )
        fest2 = Festival(
            name="День города Черняховск",
            start_date="2024-09-05",
            end_date="2024-09-07",
            city="Черняховск",
            location_name="Черняховск",
        )
        session.add_all([fest1, fest2])
        await session.commit()

    counter = 0

    async def fake_create_page(tg, *args, **kwargs):
        nonlocal counter
        counter += 1
        return {"path": f"path{counter}", "url": f"http://t.me/path{counter}"}

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None, domain=None: object())

    async def fake_build(*a, **k):
        return "", "", 0

    monkeypatch.setattr(main, "build_source_page_content", fake_build)

    async with db.get_session() as session:
        fest1 = await session.get(Festival, fest1.id)
        fest2 = await session.get(Festival, fest2.id)
        for fest in (fest1, fest2):
            start = parse_iso_date(fest.start_date)
            end = parse_iso_date(fest.end_date)
            city_from_name = parse_city_from_fest_name(fest.name)
            city_for_days = (fest.city or city_from_name or "").strip()
            if not fest.city and city_for_days:
                fest.city = city_for_days
            for i in range((end - start).days + 1):
                day = start + timedelta(days=i)
                ev = Event(
                    title=f"{fest.name} - день {i+1}",
                    description="",
                    festival=fest.name,
                    date=day.isoformat(),
                    time="",
                    location_name=fest.location_name or "",
                    location_address=fest.location_address,
                    city=city_for_days,
                    source_text=f"{fest.name} — {day.isoformat()}",
                )
                saved, _ = await upsert_event(session, ev)
                await schedule_event_update_tasks(db, saved)
        await session.commit()

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
        zel_events = [e for e in events if e.festival == "День города Зеленоградск"]
        chr_events = [e for e in events if e.festival == "День города Черняховск"]
        assert all(e.city == "Зеленоградск" for e in zel_events)
        assert all(e.city == "Черняховск" for e in chr_events)

    for e in events:
        await main.update_telegraph_event_page(e.id, db, bot)

    async with db.get_session() as session:
        paths = [e.telegraph_path for e in (await session.execute(select(Event))).scalars()]
        assert len(paths) == len(set(paths))
