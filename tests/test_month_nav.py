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
        nav_block = await main.build_month_nav_block(db, m)
        _, content, _ = await main.build_month_page_content(db, m)
        html = main.unescape_html_comments(nodes_to_html(content))
        html = main.ensure_footer_nav_with_hr(html, nav_block, month=m, page=1)
        for other in months:
            name = main.month_name_nominative(other)
            if other == m:
                assert f'<a href="https://t.me/{other}">' not in html
                assert name in html
            else:
                assert f'<a href="https://t.me/{other}">{name}</a>' in html


@pytest.mark.asyncio
async def test_month_nav_skips_past_and_empty(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    # Events in August, September, November only
    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    title="A", description="d", source_text="s",
                    date="2025-08-26", time="18:00", location_name="Hall"
                ),
                Event(
                    title="S", description="d", source_text="s",
                    date="2025-09-02", time="18:00", location_name="Hall"
                ),
                Event(
                    title="N", description="d", source_text="s",
                    date="2025-11-01", time="18:00", location_name="Hall"
                ),
            ]
        )
        for m in ("2025-08", "2025-09", "2025-11"):
            session.add(MonthPage(month=m, url=f"https://t.me/{m}", path=m))
        await session.commit()

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 8, 26)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 8, 26, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    _, content_aug, _ = await main.build_month_page_content(db, "2025-08")
    html_aug = main.unescape_html_comments(nodes_to_html(content_aug))
    nav_block_aug = await main.build_month_nav_block(db, "2025-08")
    html_aug = main.ensure_footer_nav_with_hr(
        html_aug, nav_block_aug, month="2025-08", page=1
    )
    assert '<h4>август <a href="https://t.me/2025-09">сентябрь</a> <a href="https://t.me/2025-11">ноябрь</a></h4>' in html_aug
    assert "июль" not in html_aug
    assert "октябрь" not in html_aug

    _, content_sep, _ = await main.build_month_page_content(db, "2025-09")
    html_sep = main.unescape_html_comments(nodes_to_html(content_sep))
    nav_block_sep = await main.build_month_nav_block(db, "2025-09")
    html_sep = main.ensure_footer_nav_with_hr(
        html_sep, nav_block_sep, month="2025-09", page=1
    )
    assert '<h4><a href="https://t.me/2025-08">август</a> сентябрь <a href="https://t.me/2025-11">ноябрь</a></h4>' in html_sep
    assert "октябрь" not in html_sep

    class FakeDate2(date):
        @classmethod
        def today(cls):
            return date(2025, 9, 1)

    class FakeDatetime2(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 9, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate2)
    monkeypatch.setattr(main, "datetime", FakeDatetime2)

    nav_block2 = await main.build_month_nav_block(db, "2025-08")
    _, content_aug2, _ = await main.build_month_page_content(db, "2025-08")
    html_aug2 = main.unescape_html_comments(nodes_to_html(content_aug2))
    html_aug2 = main.ensure_footer_nav_with_hr(
        html_aug2, nav_block2, month="2025-08", page=1
    )
    assert '<a href="https://t.me/2025-08">' not in html_aug2
    assert '<h4><a href="https://t.me/2025-09">сентябрь</a> <a href="https://t.me/2025-11">ноябрь</a></h4>' in html_aug2


@pytest.mark.asyncio
async def test_month_nav_includes_festival_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await main.set_setting_value(db, "fest_index_url", "https://telegra.ph/fests")

    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 8, 26)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 8, 26, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    async with db.get_session() as session:
        session.add(
            Event(
                title="E",
                description="d",
                source_text="s",
                date="2025-08-26",
                time="18:00",
                location_name="Hall",
            )
        )
        session.add(MonthPage(month="2025-08", url="https://t.me/2025-08", path="2025-08"))
        await session.commit()

    nav_block = await main.build_month_nav_block(db, "2025-08")
    assert (
        '</h4><br/><h3><a href="https://telegra.ph/fests">Фестивали</a></h3>'
        in nav_block
    )
