import pytest
from datetime import date
from sqlalchemy import select

import main
from markup import DAY_START, DAY_END, PERM_START, PERM_END


@pytest.mark.asyncio
async def test_patch_month_page_inserts_chronologically(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    # insert month page and sample event for 15th
    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-08", url="u", path="p"))
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-15",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    html = (
        "Intro"
        + DAY_START("2025-08-14")
        + "14"
        + DAY_END("2025-08-14")
        + DAY_START("2025-08-17")
        + "17"
        + DAY_END("2025-08-17")
        + PERM_START
        + "perm"
        + PERM_END
    )

    class FakeTelegraph:
        def __init__(self):
            self.edited_html = None

        def get_page(self, path, return_html=True):
            assert path == "p"
            return {"content_html": html, "title": "Title"}

        def edit_page(self, path, title, html_content, **kwargs):
            self.edited_html = html_content
            return {"path": path}

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 15))
    assert changed is True
    result = tg.edited_html
    # ensure new day inserted before 17th and before permanent section
    assert result.index(DAY_START("2025-08-14")) < result.index(DAY_START("2025-08-15")) < result.index(DAY_START("2025-08-17"))
    assert result.index(DAY_START("2025-08-15")) < result.index(PERM_START)


@pytest.mark.asyncio
async def test_patch_month_page_handles_content_too_big(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-08", url="u", path="p"))
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-15",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    html = PERM_START + "perm" + PERM_END

    class FakeTelegraph:
        def get_page(self, path, return_html=True):
            assert path == "p"
            return {"content_html": html, "title": "Title"}

        def edit_page(self, path, title, html_content, **kwargs):
            raise main.TelegraphException("CONTENT_TOO_BIG")

    called = False

    async def fake_sync(db_obj, month_key, update_links=False, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(main, "sync_month_page", fake_sync)

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 15))
    assert changed == "rebuild"
    assert called is True
    h = await main.get_section_hash(db, "telegraph:month:2025-08", "day:2025-08-15")
    assert h is not None


@pytest.mark.asyncio
async def test_patch_month_page_handles_escaped_legacy_markers(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-08", url="u", path="p"))
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-15",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    html = (
        "Intro"
        + "<!-- DAY:2025-08-15 START -->OLD<!-- DAY:2025-08-15 END -->"
        + PERM_START
        + "perm"
        + PERM_END
    )
    html = html.replace("<!--", "&lt;!--").replace("-->", "--&gt;")

    class FakeTelegraph:
        def __init__(self):
            self.edited_html = None

        def get_page(self, path, return_html=True):
            assert path == "p"
            return {"content_html": html, "title": "Title"}

        def edit_page(self, path, title, html_content, **kwargs):
            self.edited_html = html_content
            return {"path": path}

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 15))
    assert changed is True
    result = tg.edited_html
    assert result.count(DAY_START("2025-08-15")) == 1
    assert "OLD" not in result
    assert "<!-- DAY:2025-08-15 START -->" not in result
    assert result.index(DAY_START("2025-08-15")) < result.index(PERM_START)
    assert "&lt;!--" not in result


@pytest.mark.asyncio
async def test_patch_month_page_converts_legacy_header(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(main.MonthPage(month="2025-08", url="u", path="p"))
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-15",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    header = f"<h3>🟥🟥🟥 {main.format_day_pretty(date(2025, 8, 15))} 🟥🟥🟥</h3>"
    html = header + PERM_START + "perm" + PERM_END

    class FakeTelegraph:
        def __init__(self):
            self.edited_html = None

        def get_page(self, path, return_html=True):
            assert path == "p"
            return {"content_html": html, "title": "Title"}

        def edit_page(self, path, title, html_content, **kwargs):
            self.edited_html = html_content
            return {"path": path}

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 15))
    assert changed is True
    result = tg.edited_html
    assert DAY_START("2025-08-15") in result
    assert DAY_END("2025-08-15") in result
    assert result.count(header) == 1
    assert result.index(DAY_START("2025-08-15")) < result.index(header) < result.index(
        DAY_END("2025-08-15")
    )


@pytest.mark.asyncio
async def test_patch_month_page_split_updates_second_part(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            main.MonthPage(month="2025-08", url="u1", path="p1", url2="u2", path2="p2")
        )
        session.add(
            main.Event(
                title="Concert",
                description="desc",
                date="2025-08-30",
                time="12:00",
                location_name="loc",
                source_text="src",
            )
        )
        await session.commit()

    html1 = DAY_START("2025-08-01") + "1" + DAY_END("2025-08-01")
    html2 = (
        DAY_START("2025-08-29")
        + "29"
        + DAY_END("2025-08-29")
        + PERM_START
        + "perm"
        + PERM_END
    )

    class FakeTelegraph:
        def __init__(self):
            self.edits: list[tuple[str, str]] = []

        def get_page(self, path, return_html=True):
            if path == "p1":
                return {"content_html": html1, "title": "T1"}
            assert path == "p2"
            return {"content_html": html2, "title": "T2"}

        def edit_page(self, path, title, html_content, **kwargs):
            self.edits.append((path, html_content))
            return {"path": path}

    tg = FakeTelegraph()
    changed = await main.patch_month_page_for_date(db, tg, "2025-08", date(2025, 8, 30))
    assert changed is True
    assert tg.edits[0][0] == "p2"
    result = tg.edits[0][1]
    assert DAY_START("2025-08-30") in result
    assert DAY_END("2025-08-30") in result

    async with db.get_session() as session:
        page = await session.get(main.MonthPage, "2025-08")
        evs = (
            await session.execute(
                select(main.Event).where(main.Event.date.like("2025-08-30%"))
            )
        ).scalars().all()
        assert page.content_hash2 == main.content_hash(result)
        expected_section = main.render_month_day_section(date(2025, 8, 30), evs)

    h = await main.get_section_hash(
        db, "telegraph:month:2025-08", "day:2025-08-30"
    )
    assert h == main.content_hash(expected_section)
