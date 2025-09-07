from pathlib import Path
from datetime import date, datetime

import pytest
import main
from models import Event
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_weekend_page_camera_icons(tmp_path: Path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 10)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 10, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    title="No",
                    description="d",
                    source_text="s",
                    date="2025-07-12",
                    time="12:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/no",
                    photo_count=0,
                ),
                Event(
                    title="One",
                    description="d",
                    source_text="s",
                    date="2025-07-12",
                    time="13:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/one",
                    photo_count=1,
                ),
                Event(
                    title="Many",
                    description="d",
                    source_text="s",
                    date="2025-07-12",
                    time="14:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/many",
                    photo_count=5,
                ),
            ]
        )
        await session.commit()

    _, content, _ = await main.build_weekend_page_content(db, "2025-07-12")
    html = nodes_to_html(content)

    assert '<a href="https://t.me/no">подробнее</a>' in html
    assert '📸 <a href="https://t.me/no">подробнее</a>' not in html
    assert '📸📸 <a href="https://t.me/no">подробнее</a>' not in html

    assert '📸 <a href="https://t.me/one">подробнее</a>' in html
    assert '📸📸 <a href="https://t.me/one">подробнее</a>' not in html

    assert '📸📸 <a href="https://t.me/many">подробнее</a>' in html
    assert '📸📸📸 <a href="https://t.me/many">подробнее</a>' not in html


@pytest.mark.asyncio
async def test_month_page_camera_icons(tmp_path: Path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 7, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 7, 1, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    title="No",
                    description="d",
                    source_text="s",
                    date="2025-07-01",
                    time="12:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/no",
                    photo_count=0,
                ),
                Event(
                    title="One",
                    description="d",
                    source_text="s",
                    date="2025-07-02",
                    time="13:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/one",
                    photo_count=1,
                ),
                Event(
                    title="Many",
                    description="d",
                    source_text="s",
                    date="2025-07-03",
                    time="14:00",
                    location_name="Loc",
                    telegraph_url="https://t.me/many",
                    photo_count=4,
                ),
            ]
        )
        await session.commit()

    _, content, _ = await main.build_month_page_content(db, "2025-07")
    html = nodes_to_html(content)

    assert '<a href="https://t.me/no">подробнее</a>' in html
    assert '📸 <a href="https://t.me/no">подробнее</a>' not in html
    assert '📸📸 <a href="https://t.me/no">подробнее</a>' not in html

    assert '📸 <a href="https://t.me/one">подробнее</a>' in html
    assert '📸📸 <a href="https://t.me/one">подробнее</a>' not in html

    assert '📸📸 <a href="https://t.me/many">подробнее</a>' in html
    assert '📸📸📸 <a href="https://t.me/many">подробнее</a>' not in html
