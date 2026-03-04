from pathlib import Path
from datetime import datetime

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


def _node_text(node) -> str:
    children = node.get("children") or []
    parts: list[str] = []
    for ch in children:
        if isinstance(ch, str):
            parts.append(ch)
            continue
        if isinstance(ch, dict):
            nested = ch.get("children") or []
            for sub in nested:
                if isinstance(sub, str):
                    parts.append(sub)
    return "".join(parts)


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
    assert len(nodes) == 4
    assert any(n.get("tag") == "p" and "📅" in _node_text(n) for n in nodes)
    assert any(n.get("tag") == "p" and "📍" in _node_text(n) for n in nodes)
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
    assert len(nodes) == 5
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
    assert any(n.get("tag") == "p" and "📅" in _node_text(n) for n in nodes)
    assert any(n.get("tag") == "p" and "📍" in _node_text(n) for n in nodes)
    assert _p_has_content(nodes[2])


def test_event_title_prefers_telegraph_link():
    fest = Festival(name="Fest")
    e = Event(
        title="Day 1",
        description="",
        source_text="s",
        date="2030-01-01",
        time="10:00",
        location_name="Loc",
        festival="Fest",
        telegraph_path="ev",
        source_post_url="https://t.me/post",
        added_at=datetime(2000, 1, 1),
    )
    nodes = main.event_to_nodes(e, festival=fest, show_festival=False)
    link = nodes[0]["children"][0]
    assert link["attrs"]["href"] == "https://telegra.ph/ev"


def test_event_title_uses_source_post_without_telegraph():
    fest = Festival(name="Fest")
    e = Event(
        title="Day 1",
        description="",
        source_text="s",
        date="2030-01-01",
        time="10:00",
        location_name="Loc",
        festival="Fest",
        source_post_url="https://t.me/post",
        added_at=datetime(2000, 1, 1),
    )
    nodes = main.event_to_nodes(e, festival=fest, show_festival=False)
    link = nodes[0]["children"][0]
    assert link["attrs"]["href"] == "https://t.me/post"


def test_event_to_nodes_invalid_html_fallback_preserves_detail_lines(monkeypatch):
    def _raise_invalid(_html):
        raise ValueError("invalid")

    monkeypatch.setattr("telegraph.utils.html_to_nodes", _raise_invalid)

    e = Event(
        title="Fallback test",
        description="Полный текст описания.",
        source_text="s",
        date="2030-01-01",
        time="10:00",
        location_name="Loc",
        location_address="Street 1",
        city="Kaliningrad",
        telegraph_url="https://telegra.ph/test",
        ics_url="https://example.com/test.ics",
        festival="Fest",
        search_digest="Короткий дайджест события.",
    )
    nodes = main.event_to_nodes(e, show_festival=False)

    p_texts: list[str] = []
    for node in nodes:
        if node.get("tag") != "p":
            continue
        children = node.get("children") or []
        parts: list[str] = []
        for ch in children:
            if isinstance(ch, str):
                parts.append(ch)
            elif isinstance(ch, dict):
                nested = ch.get("children") or []
                for sub in nested:
                    if isinstance(sub, str):
                        parts.append(sub)
        txt = "".join(parts).strip()
        if txt:
            p_texts.append(txt)

    assert any(t.startswith("📅 ") for t in p_texts)
    assert any(t.startswith("📍 ") for t in p_texts)
    assert any("подробнее" in t for t in p_texts)
    digest_line = next((t for t in p_texts if "дайджест" in t.lower()), "")
    assert digest_line
    assert "📅" not in digest_line
    assert "📍" not in digest_line

    links: list[tuple[str, str]] = []
    for node in nodes:
        if node.get("tag") != "p":
            continue
        for ch in node.get("children") or []:
            if not isinstance(ch, dict) or ch.get("tag") != "a":
                continue
            href = str((ch.get("attrs") or {}).get("href") or "").strip()
            label = "".join([s for s in (ch.get("children") or []) if isinstance(s, str)]).strip()
            links.append((label, href))
    assert any("подробнее" in label.lower() and href == "https://telegra.ph/test" for label, href in links)
    assert any("добавить в календарь" in label.lower() and href == "https://example.com/test.ics" for label, href in links)


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


@pytest.mark.asyncio
async def test_build_festival_page_content_event_gallery_order(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FESTIVALS_UPCOMING_HORIZON_DAYS", "3650")
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            description="d",
            photo_urls=[
                "https://example.com/cover.jpg",
                "https://example.com/gallery.jpg",
            ],
        )

        ev = Event(
            title="Day 1",
            description="",
            source_text="s",
            date="2030-01-01",
            time="10:00",
            location_name="Loc",
            festival="Fest",
            telegraph_path="ev",
            photo_urls=["https://example.com/img.jpg"],
        )
        other = Festival(
            name="Other",
            description="o",
            start_date="2030-01-05",
            end_date="2030-01-06",
        )
        session.add(fest)
        session.add(ev)
        session.add(other)
        await session.commit()
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert '<img src="https://example.com/img.jpg"' not in html
    assert '<a href="https://telegra.ph/ev">Day 1</a>' in html
    cover_idx = html.index("https://example.com/cover.jpg")
    events_idx = html.index("Мероприятия фестиваля")
    gallery_idx = html.index("https://example.com/gallery.jpg")
    nav_idx = html.index("Ближайшие фестивали")
    assert cover_idx < events_idx < gallery_idx < nav_idx
