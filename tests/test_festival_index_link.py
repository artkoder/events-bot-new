import pytest
from pathlib import Path
import logging

import main
from db import Database
from models import Festival
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_festival_page_has_index_link(tmp_path: Path, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await main.set_setting_value(db, "fest_index_url", "https://telegra.ph/fests")

    async with db.get_session() as session:
        fest = Festival(name="Fest")
        session.add(fest)
        await session.commit()

    with caplog.at_level(logging.INFO):
        _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert (
        '<a href="https://telegra.ph/fests">🎪 Все фестивали Калининградской области →</a>'
        in html
    )
    rec = next(r for r in caplog.records if r.message == "festival_page_index_link")
    assert rec.festival == "Fest"
    assert rec.fest_index_url == "https://telegra.ph/fests"
