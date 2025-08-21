import pytest
from pathlib import Path
from datetime import date, datetime

import main
from db import Database
from models import Event, MonthPage
from telegraph.utils import nodes_to_html

@pytest.mark.asyncio
async def test_footer_links_propagate_across_all_month_pages(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Existing months: August, September, October
    async with db.get_session() as session:
        for month in ("2025-08", "2025-09", "2025-10"):
            session.add(
                Event(
                    title="E",
                    description="d",
                    source_text="s",
                    date=f"{month}-10",
                    time="18:00",
                    location_name="Hall",
                )
            )
            session.add(MonthPage(month=month, url=f"https://t.me/{month}", path=month))
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 10)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 10, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    # Add November month and event
    async with db.get_session() as session:
        session.add(
            Event(
                title="E",
                description="d",
                source_text="s",
                date="2025-11-10",
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(MonthPage(month="2025-11", url="https://t.me/2025-11", path="2025-11"))
        await session.commit()

    months = ["2025-08", "2025-09", "2025-10", "2025-11"]
    for m in months:
        _, content, _ = await main.build_month_page_content(db, m)
        html = nodes_to_html(content)
        for other in months:
            name = main.month_name_nominative(other)
            if other == m:
                assert name in html
            else:
                assert f'<a href="https://t.me/{other}">{name}</a>' in html
