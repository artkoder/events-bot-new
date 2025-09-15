import pytest
from pathlib import Path

import main
from db import Database
from models import Event, Festival


@pytest.mark.asyncio
async def test_source_page_has_festival_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = Festival(name="Fest", telegraph_path="fest")
        ev = Event(
            title="T",
            description="d",
            source_text="src text",
            date="2025-07-16",
            time="18:00",
            location_name="Hall",
            festival=fest.name,
        )
        session.add_all([fest, ev])
        await session.commit()

    async def fake_nav(db):
        return "<p>NAV</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_nav)

    html, _, _ = await main.build_source_page_content(
        "T", "src text", None, None, None, None, db
    )
    snippet = '<p>&#8203;</p><p>✨ <a href="https://telegra.ph/fest">Fest</a></p><p>&#8203;</p>'
    assert snippet in html
    assert html.index('✨') < html.index('<p>NAV</p>')
