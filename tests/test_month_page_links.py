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
