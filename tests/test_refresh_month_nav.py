import pytest
from datetime import date, datetime


@pytest.mark.asyncio
async def test_refresh_month_nav_creates_missing_month_page(tmp_path, monkeypatch):
    import main

    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Freeze time to September 2025
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 9, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 9, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    # Prepare DB: September page exists, December page missing
    async with db.get_session() as session:
        session.add(
            main.MonthPage(month="2025-09", url="https://t.me/sep", path="sep")
        )
        session.add(
            main.Event(
                title="S", description="d", source_text="s",
                date="2025-09-10", time="18:00", location_name="Hall"
            )
        )
        session.add(
            main.Event(
                title="D", description="d", source_text="s",
                date="2025-12-05", time="18:00", location_name="Hall"
            )
        )
        await session.commit()

    calls: list[tuple[str, bool]] = []

    async def fake_sync_month_page(db_obj, month: str, update_links: bool = False, force: bool = False):
        calls.append((month, update_links))
        async with db_obj.get_session() as sess:
            page = await sess.get(main.MonthPage, month)
            if not page:
                page = main.MonthPage(month=month, url=f"https://t.me/{month}", path=month)
                sess.add(page)
            elif not page.url:
                page.url = f"https://t.me/{month}"
                page.path = month
            await sess.commit()
        if not update_links:
            await main.refresh_month_nav(db_obj)

    monkeypatch.setattr(main, "sync_month_page", fake_sync_month_page)

    # Simulate updating September page which should trigger navigation refresh
    await main.sync_month_page(db, "2025-09")

    # December page should be created automatically
    async with db.get_session() as session:
        dec_page = await session.get(main.MonthPage, "2025-12")

    assert dec_page is not None and dec_page.url

    # Ensure recursive calls were made as expected
    assert ("2025-12", False) in calls
    assert ("2025-09", True) in calls
    assert ("2025-12", True) in calls

    # Navigation helper should include both months
    nav_html = await main.build_month_nav_html(db)
    assert dec_page.url in nav_html

