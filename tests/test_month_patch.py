from datetime import date

import pytest

import main
from models import Event, MonthPage
from markup import PERM_START, PERM_END


class FakeTelegraph:
    def __init__(self, html: str, title: str = "Title"):
        self.html = html
        self.title = title
        self.edited_html: str | None = None
        self.edits: list[tuple[str, str]] = []

    def get_page(self, path, return_html=True):
        return {"content_html": self.html, "title": self.title}

    def edit_page(self, path, title=None, html_content=None, **kwargs):
        self.edited_html = html_content
        self.edits.append((path, html_content))
        return {"path": path}


@pytest.mark.asyncio
async def test_patch_adds_links_and_idempotent(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-09", url="u", path="p"))
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-08",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        html = main.render_month_day_section(date(2025, 9, 8), [ev])

    tg = FakeTelegraph(html)

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        ev.telegraph_url = "https://t.me/test"
        ev.ics_post_url = "https://t.me/file"
        await session.commit()

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 8))
    assert changed is True
    assert "подробнее" in tg.edited_html and "https://t.me/test" in tg.edited_html
    assert "Добавить в календарь" in tg.edited_html and "https://t.me/file" in tg.edited_html

    tg.edited_html = None
    changed2 = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 8))
    assert changed2 is False
    assert tg.edited_html is None


@pytest.mark.asyncio
async def test_patch_inserts_missing_section(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-09", url="u", path="p"))
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-08",
                time="12:00",
                location_name="loc",
                source_text="src",
                telegraph_url="https://t.me/test",
                ics_url="https://sup.ics",
            )
        )
        await session.commit()

    html = PERM_START + "perm" + PERM_END
    tg = FakeTelegraph(html)

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 8))
    assert changed is True
    assert "<h3>🟥🟥🟥 8 сентября 🟥🟥🟥</h3>" in tg.edited_html
    assert "https://t.me/test" in tg.edited_html
    assert "https://sup.ics" in tg.edited_html


@pytest.mark.asyncio
async def test_patch_split_updates_second_part(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            MonthPage(month="2025-09", url="u1", path="p1", url2="u2", path2="p2")
        )
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-30",
                time="12:00",
                location_name="loc",
                source_text="src",
                telegraph_url="https://t.me/test",
                ics_url="https://sup.ics",
            )
        )
        await session.commit()

    html1 = main.render_month_day_section(date(2025, 9, 1), [])
    html2 = PERM_START + "perm" + PERM_END
    tg = FakeTelegraph("", title="")

    def get_page(path, return_html=True):
        if path == "p1":
            return {"content_html": html1, "title": "T1"}
        assert path == "p2"
        return {"content_html": html2, "title": "T2"}

    tg.get_page = get_page

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 30))
    assert changed is True
    assert tg.edits[0][0] == "p2"
    assert "https://t.me/test" in tg.edits[0][1]
    assert "https://sup.ics" in tg.edits[0][1]


@pytest.mark.asyncio
async def test_patch_replace_keeps_position(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-09", url="u", path="p"))
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-27",
                time="12:00",
                location_name="loc",
                source_text="src",
                telegraph_url="https://t.me/test",
                ics_url="https://sup.ics",
            )
        )
        await session.commit()

    html = (
        main.render_month_day_section(date(2025, 9, 26), [])
        + main.render_month_day_section(date(2025, 9, 27), [])
        + main.render_month_day_section(date(2025, 9, 28), [])
        + "<hr><p>nav</p>"
    )
    tg = FakeTelegraph(html)

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 27))
    assert changed is True
    edited = tg.edited_html
    assert edited.count("27 сентября") == 1
    assert edited.index("26 сентября") < edited.index("27 сентября") < edited.index("28 сентября")
    tail = edited.split("<hr>")[-1]
    assert "<h3>" not in tail

    tg.edited_html = None
    changed2 = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 27))
    assert changed2 is False
    assert tg.edited_html is None


@pytest.mark.asyncio
async def test_patch_insert_sorted_between_dates(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-09", url="u", path="p"))
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-27",
                time="12:00",
                location_name="loc",
                source_text="src",
                telegraph_url="https://t.me/test",
                ics_url="https://sup.ics",
            )
        )
        await session.commit()

    html = (
        main.render_month_day_section(date(2025, 9, 26), [])
        + main.render_month_day_section(date(2025, 9, 28), [])
        + "<hr><p>nav</p>"
    )
    tg = FakeTelegraph(html)

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 27))
    assert changed is True
    edited = tg.edited_html
    assert edited.index("26 сентября") < edited.index("27 сентября") < edited.index("28 сентября")
    tail = edited.split("<hr>")[-1]
    assert "<h3>" not in tail

    tg.edited_html = None
    changed2 = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 27))
    assert changed2 is False
    assert tg.edited_html is None


@pytest.mark.asyncio
async def test_patch_insert_before_hr(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(MonthPage(month="2025-09", url="u", path="p"))
        session.add(
            Event(
                title="Concert",
                description="desc",
                date="2025-09-30",
                time="12:00",
                location_name="loc",
                source_text="src",
                telegraph_url="https://t.me/test",
                ics_url="https://sup.ics",
            )
        )
        await session.commit()

    html = (
        main.render_month_day_section(date(2025, 9, 28), [])
        + "<hr><p>nav</p>"
    )
    tg = FakeTelegraph(html)

    changed = await main.patch_month_page_for_date(db, tg, "2025-09", date(2025, 9, 30))
    assert changed is True
    edited = tg.edited_html
    assert "28 сентября" in edited and "30 сентября" in edited
    assert edited.index("28 сентября") < edited.index("30 сентября")
    tail = edited.split("<hr>")[-1]
    assert "<h3>" not in tail
