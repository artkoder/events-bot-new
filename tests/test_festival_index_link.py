import pytest
from pathlib import Path

import main
from db import Database
from models import Festival
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_festival_page_has_index_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await main.set_setting_value(db, "fest_index_url", "https://telegra.ph/fests")

    async with db.get_session() as session:
        fest = Festival(name="Fest")
        session.add(fest)
        await session.commit()

    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "Все фестивали Калининградской области" in html
    assert "https://telegra.ph/fests" in html
