from pathlib import Path

import pytest

import main
from db import Database
from models import Event, Festival
from telegraph.utils import nodes_to_html


def _p_has_content(node):
    if node.get("tag") != "p":
        return True
    children = node.get("children") or []
    for ch in children:
        if isinstance(ch, str) and ch.strip():
            return True
        if isinstance(ch, dict) and ch.get("tag") != "br":
            return True
    return False


def test_event_to_nodes_autoday_no_program():
    fest = Festival(name="Fest")
    e = Event(
        title="Day 1",
        description="",
        source_text="s",
        date="2030-01-01",
        time="10:00",
        location_name="Loc",
        festival="Fest",
    )
    nodes = main.event_to_nodes(e, festival=fest, show_festival=False)
    assert len(nodes) == 3
    assert _p_has_content(nodes[1])


def test_event_to_nodes_autoday_program_link():
    fest = Festival(name="Fest", program_url="https://prog")
    e = Event(
        title="Day 1",
        description="",
        source_text="s",
        date="2030-01-01",
        time="10:00",
        location_name="Loc",
        festival="Fest",
    )
    nodes = main.event_to_nodes(e, festival=fest, show_festival=False)
    assert len(nodes) == 4
    assert nodes[1] == {
        "tag": "p",
        "children": [
            {
                "tag": "a",
                "attrs": {"href": "https://prog"},
                "children": ["программа"],
            }
        ],
    }
    assert _p_has_content(nodes[2])


@pytest.mark.asyncio
async def test_build_festival_page_content_autoday(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(name="Fest", program_url="https://prog", description="d")
        ev = Event(
            title="Day 1",
            description="",
            source_text="s",
            date="2030-01-01",
            time="10:00",
            location_name="Loc",
            festival="Fest",
        )
        session.add(fest)
        session.add(ev)
        await session.commit()
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert '<a href="https://prog">программа</a>' in html
    assert '<p></p>' not in html


@pytest.mark.asyncio
async def test_build_festival_page_content_autoday_no_program(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(name="Fest", description="d")
        ev = Event(
            title="Day 1",
            description="",
            source_text="s",
            date="2030-01-01",
            time="10:00",
            location_name="Loc",
            festival="Fest",
        )
        session.add(fest)
        session.add(ev)
        await session.commit()
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert '<a href="https://prog">программа</a>' not in html
    assert '<p></p>' not in html
