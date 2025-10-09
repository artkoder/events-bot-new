from pathlib import Path

import pytest

import main
from db import Database
from models import Festival
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_festival_page_program_block(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            program_url="https://prog",
            website_url="https://site",
            activities_json=[{"title": "Opening", "time": "10:00"}],
        )
        session.add(fest)
        await session.commit()

    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert '<h2>ПРОГРАММА</h2>' in html
    assert '<a href="https://prog">Смотреть программу</a>' in html
    assert '<a href="https://site">Сайт</a>' in html

