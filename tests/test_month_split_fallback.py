from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import main
from db import Database
from models import Event, MonthPage


@pytest.mark.asyncio
async def test_split_month_until_ok_fallback_without_ics(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    month = "2025-07"

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2025, 7, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2025, 7, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        page = MonthPage(month=month, url="", path="")
        session.add(page)
        await session.commit()

    events: list[Event] = []
    for idx in range(4):
        events.append(
            Event(
                title=f"Event {idx}",
                description="Description",
                source_text="Source text",
                date=f"{month}-{10 + idx:02d}",
                time="18:00",
                location_name="Hall",
                telegraph_url=f"https://telegra.ph/event{idx}",
                ics_url=f"https://example.com/event{idx}.ics",
                added_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            )
        )

    nav_block = "<nav>links</nav>"

    create_attempts: list[str] = []
    success_htmls: list[str] = []

    async def fake_create_page(
        tg,
        *,
        title,
        html_content,
        caller="event_pipeline",
        **kwargs,
    ):
        lower = html_content.lower()
        create_attempts.append(lower)
        if "добавить в календарь" in lower:
            raise main.TelegraphException("CONTENT TOO BIG")
        success_htmls.append(lower)
        idx = len(success_htmls)
        return {"url": f"https://telegra.ph/page{idx}", "path": f"path{idx}"}

    edit_attempts: list[str] = []

    async def fake_edit_page(
        tg,
        path,
        *,
        title,
        html_content,
        caller="event_pipeline",
        **kwargs,
    ):
        lower = html_content.lower()
        edit_attempts.append(lower)
        if "добавить в календарь" in lower:
            raise main.TelegraphException("CONTENT TOO BIG")
        return {"url": f"https://telegra.ph/{path}", "path": path}

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)

    original_build = main.build_month_page_content
    include_flags: list[tuple[bool, bool]] = []

    async def tracked_build_month_page_content(
        db_obj,
        month_str,
        events_list,
        exhibitions_list,
        continuation_url=None,
        size_limit=None,
        *,
        include_ics=True,
        include_details=True,
    ):
        include_flags.append((include_ics, include_details))
        return await original_build(
            db_obj,
            month_str,
            events_list,
            exhibitions_list,
            continuation_url=continuation_url,
            size_limit=size_limit,
            include_ics=include_ics,
            include_details=include_details,
        )

    monkeypatch.setattr(main, "build_month_page_content", tracked_build_month_page_content)

    tg = object()

    page_obj: MonthPage
    async with db.get_session() as session:
        page_obj = await session.get(MonthPage, month)

    await main.split_month_until_ok(db, tg, page_obj, month, events, [], nav_block)

    assert any("добавить в календарь" in html for html in create_attempts)
    assert all("добавить в календарь" not in html for html in success_htmls)
    assert include_flags[-1] == (False, True)
    assert any(not flags[0] for flags in include_flags)

    async with db.get_session() as session:
        stored = await session.get(MonthPage, month)

    assert stored is not None
    assert stored.path
    assert stored.path2

