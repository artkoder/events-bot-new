import importlib
from datetime import date

import pytest
import main as orig_main


@pytest.mark.asyncio
async def test_month_page_updated_after_telegraph_build(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)

    monkeypatch.setattr(m, "get_telegraph_token", lambda: "t")

    async def fake_location(parts):
        return " ".join(part.strip() for part in parts if part)

    monkeypatch.setattr(m, "build_short_vk_location", fake_location)

    counter = {"i": 0}

    async def fake_create_page(tg, *args, **kwargs):
        counter["i"] += 1
        return {"url": f"https://tg/{counter['i']}", "path": f"pg{counter['i']}"}

    monkeypatch.setattr(m, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(m, "update_source_post_keyboard", lambda *a, **k: None)

    class FakeTelegraph:
        pages = {"p": ""}

        def __init__(self, access_token=None):
            pass

        def get_page(self, path, return_html=True):
            return {"content": FakeTelegraph.pages.get(path, ""), "title": "Title"}

        def edit_page(self, path, title, html_content):
            FakeTelegraph.pages[path] = html_content
            return {"path": path}

    monkeypatch.setattr(m, "Telegraph", FakeTelegraph)

    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(m.MonthPage(month="2025-09", url="u", path="p"))
        e1 = m.Event(
            title="E1",
            description="d",
            date="2025-09-09",
            time="13:00",
            location_name="Loc",
            city="Town",
            source_text="SRC",
        )
        e2 = m.Event(
            title="E2",
            description="d",
            date="2025-09-16",
            time="17:00",
            location_name="Loc",
            city="Town",
            source_text="SRC",
        )
        session.add_all([e1, e2])
        await session.commit()
        ids = [e1.id, e2.id]

    for eid in ids:
        await m.update_month_pages_for(eid, db, None)

    assert "Подробнее" not in FakeTelegraph.pages["p"]

    for eid in ids:
        await m.update_telegraph_event_page(eid, db, None)

    html = FakeTelegraph.pages["p"]
    assert html.count("подробнее") == 2
    assert "https://tg/1" in html
    assert "https://tg/2" in html


@pytest.mark.asyncio
async def test_split_month_until_ok_strips_details_when_needed(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)

    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    month = "2025-08"

    async with db.get_session() as session:
        session.add(m.MonthPage(month=month, url="", path=""))
        await session.commit()

    events = []
    for idx in range(4):
        events.append(
            m.Event(
                title=f"Event {idx}",
                description="Description",
                source_text="Source text",
                date=f"{month}-{10 + idx:02d}",
                time="18:00",
                location_name="Hall",
                telegraph_url=f"https://telegra.ph/event{idx}",
                ics_url=f"https://example.com/event{idx}.ics",
            )
        )

    combos = []
    original_build = m.build_month_page_content

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
        combos.append((include_ics, include_details))
        result = await original_build(
            db_obj,
            month_str,
            events_list,
            exhibitions_list,
            continuation_url=continuation_url,
            size_limit=size_limit,
            include_ics=include_ics,
            include_details=include_details,
        )
        if include_ics or include_details:
            raise m.TelegraphException("CONTENT_TOO_BIG")
        return result

    monkeypatch.setattr(m, "build_month_page_content", tracked_build_month_page_content)

    created_html = []

    async def fake_create_page(tg, *, title, html_content, caller="event_pipeline", **kwargs):
        created_html.append(html_content.lower())
        idx = len(created_html)
        return {"url": f"https://telegra.ph/page{idx}", "path": f"path{idx}"}

    async def fake_edit_page(
        tg,
        path,
        *,
        title,
        html_content,
        caller="event_pipeline",
        **kwargs,
    ):
        created_html.append(html_content.lower())
        return {"url": f"https://telegra.ph/{path}", "path": path}

    monkeypatch.setattr(m, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(m, "telegraph_edit_page", fake_edit_page)

    tg = object()
    nav_block = "<nav>links</nav>"

    async with db.get_session() as session:
        page_obj = await session.get(m.MonthPage, month)

    await m.split_month_until_ok(db, tg, page_obj, month, events, [], nav_block)

    assert (True, True) in combos
    assert (False, True) in combos
    assert combos[-1] == (False, False)
    assert created_html, "expected successful html content"
    assert all("подробнее" not in html for html in created_html)
    assert all("добавить в календарь" not in html for html in created_html)
