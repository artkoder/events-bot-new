from pathlib import Path

import pytest

import main
from db import Database
from models import Festival


@pytest.mark.asyncio
async def test_festival_cover_comes_first(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            photo_url="https://example.com/cover.jpg",
            program_url="https://prog",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)

    _, nodes = await main.build_festival_page_content(db, fest)

    assert nodes[0]["tag"] == "figure"
    img = next(
        ch for ch in nodes[0].get("children", []) if isinstance(ch, dict) and ch.get("tag") == "img"
    )
    assert img["attrs"]["src"] == "https://example.com/cover.jpg"
    h2_idx = next(i for i, n in enumerate(nodes) if n.get("tag") == "h2")
    assert h2_idx > 0

