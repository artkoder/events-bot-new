import asyncio
from datetime import date, timedelta, datetime
from pathlib import Path

import pytest
from aiogram import Bot
from sqlmodel import select

import main
from main import Database, Event, Festival, upsert_event, schedule_event_update_tasks, parse_iso_date
from telegraph.utils import nodes_to_html


class DummyBot(Bot):
    async def request(self, method, data=None, files=None):  # pragma: no cover - network stub
        return None


@pytest.mark.asyncio
async def test_weekend_page_two_festdays(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest1 = Festival(
            name="День города Зеленоградск",
            start_date="2025-09-06",
            end_date="2025-09-07",
            city="Зеленоградск",
            location_name="Зеленоградск",
        )
        fest2 = Festival(
            name="День города Черняховск",
            start_date="2025-09-05",
            end_date="2025-09-07",
            city="Черняховск",
            location_name="Черняховск",
        )
        session.add_all([fest1, fest2])
        await session.commit()

    async def fake_create_page(tg, *args, **kwargs):
        return {"path": "p", "url": "http://t.me/p"}

    async def fake_edit_page(tg, path, **kwargs):
        return None

    async def fake_build(*a, **k):
        return "<p>src</p>", [], 0

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)
    monkeypatch.setattr(main, "build_source_page_content", fake_build)
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None, domain=None: object())
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")

    async with db.get_session() as session:
        f1 = await session.get(Festival, fest1.id)
        f2 = await session.get(Festival, fest2.id)
        for fest in (f1, f2):
            start = parse_iso_date(fest.start_date)
            end = parse_iso_date(fest.end_date)
            for i in range((end - start).days + 1):
                day = start + timedelta(days=i)
                ev = Event(
                    title=f"{fest.name} - день {i+1}",
                    description="d",
                    festival=fest.name,
                    date=day.isoformat(),
                    time="12:00",
                    location_name=fest.location_name or "",
                    city=fest.city,
                    source_text=f"{fest.name} — {day.isoformat()}",
                )
                saved, _ = await upsert_event(session, ev)
                await schedule_event_update_tasks(db, saved, drain_nav=False)
        await session.commit()

    bot = DummyBot("1:1")
    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()
    for e in events:
        await main.update_telegraph_event_page(e.id, db, bot)

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 9, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 9, 1, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    _, content, _ = await main.build_weekend_page_content(db, "2025-09-06")
    html = nodes_to_html(content)

    async with db.get_session() as session:
        evs = (await session.execute(select(Event))).scalars().all()
    zel_url = next(e.telegraph_url for e in evs if e.festival == "День города Зеленоградск" and e.date == "2025-09-06")
    chr_url = next(e.telegraph_url for e in evs if e.festival == "День города Черняховск" and e.date == "2025-09-06")

    assert "#Зеленоградск" in html
    assert "#Черняховск" in html
    assert zel_url in html
    assert chr_url in html
