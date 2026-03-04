from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from aiogram import types
from sqlalchemy import select

import main
from db import Database
from models import Event, EventSource, Festival, TelegramSource
from telegraph.utils import nodes_to_html


@pytest.mark.asyncio
async def test_ensure_festival_merges_photo_urls(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def nop(*a, **k):
        return None

    monkeypatch.setattr(main, "sync_festival_page", nop)
    monkeypatch.setattr(main, "sync_festival_vk_post", nop)
    monkeypatch.setattr(main, "sync_festivals_index_page", nop)
    monkeypatch.setattr(main, "notify_superadmin", nop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", nop)

    urls1 = [f"https://catbox.moe/{i}.jpg" for i in range(3)]
    fest, created, updated = await main.ensure_festival(db, "Fest", photo_urls=urls1)
    assert created and updated
    assert fest.photo_url == urls1[0]
    assert fest.photo_urls == urls1

    urls2 = [urls1[1], "https://catbox.moe/new.jpg"]
    fest2, created2, updated2 = await main.ensure_festival(db, "Fest", photo_urls=urls2)
    assert not created2 and updated2
    assert fest2.photo_urls == urls1 + ["https://catbox.moe/new.jpg"]
    assert fest2.photo_url == urls1[0]


@pytest.mark.asyncio
async def test_ensure_festival_does_not_overwrite_photo_url(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def nop(*a, **k):
        return None

    monkeypatch.setattr(main, "sync_festival_page", nop)
    monkeypatch.setattr(main, "sync_festival_vk_post", nop)
    monkeypatch.setattr(main, "sync_festivals_index_page", nop)
    monkeypatch.setattr(main, "notify_superadmin", nop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", nop)

    first = "https://catbox.moe/first.jpg"
    fest, created, updated = await main.ensure_festival(
        db, "Fest", photo_url=first, photo_urls=[first]
    )
    assert created and updated
    assert fest.photo_url == first

    second = "https://catbox.moe/second.jpg"
    fest2, created2, updated2 = await main.ensure_festival(
        db, "Fest", photo_url=second, photo_urls=[second]
    )
    assert not created2 and updated2
    assert fest2.photo_url == first
    assert fest2.photo_urls == [first, second]


@pytest.mark.asyncio
async def test_ensure_festival_updates_urls(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def nop(*a, **k):
        return None

    monkeypatch.setattr(main, "sync_festival_page", nop)
    monkeypatch.setattr(main, "sync_festival_vk_post", nop)
    monkeypatch.setattr(main, "sync_festivals_index_page", nop)
    monkeypatch.setattr(main, "notify_superadmin", nop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", nop)

    fest, created, updated = await main.ensure_festival(
        db,
        "Fest",
        website_url=" https://site ",
        program_url="https://prog",
        ticket_url="https://tickets",
    )

    assert created and updated
    assert fest.website_url == "https://site"
    assert fest.program_url == "https://prog"
    assert fest.ticket_url == "https://tickets"

    fest2, created2, updated2 = await main.ensure_festival(
        db,
        "Fest",
        website_url="https://site",
        program_url="https://prog2",
        ticket_url="https://tickets2",
    )

    assert not created2 and updated2
    assert fest2.website_url == "https://site"
    assert fest2.program_url == "https://prog2"
    assert fest2.ticket_url == "https://tickets2"

    fest3, created3, updated3 = await main.ensure_festival(
        db,
        "Fest",
        program_url="https://prog2",
    )

    assert not created3 and not updated3
    assert fest3.program_url == "https://prog2"


@pytest.mark.asyncio
async def test_build_festival_page_content_shows_album(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    urls = [f"https://example.com/{i}.jpg" for i in range(3)]
    async with db.get_session() as session:
        fest = Festival(name="Fest", photo_url=urls[1], photo_urls=urls, description="desc")
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)

    def _collect_img_srcs(nodes):
        srcs = []
        for n in nodes:
            if n.get("tag") == "img":
                srcs.append(n["attrs"]["src"])
            elif n.get("tag") == "figure":
                for ch in n.get("children", []):
                    if isinstance(ch, dict) and ch.get("tag") == "img":
                        srcs.append(ch["attrs"]["src"])
        return srcs

    srcs = _collect_img_srcs(nodes)
    assert srcs == [urls[1], urls[0], urls[2]]


@pytest.mark.asyncio
async def test_build_festival_page_content_skips_catbox_images(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    safe_cover = "https://example.com/safe-cover.jpg"
    catbox_cover = "https://files.catbox.moe/a.jpg"
    catbox_gallery = "https://files.catbox.moe/b.jpg"
    safe_gallery = "https://example.com/safe-gallery.jpg"
    await main.set_setting_value(db, "festivals_index_cover", safe_cover)

    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            photo_url=catbox_cover,
            photo_urls=[catbox_cover, catbox_gallery, safe_gallery],
            description="desc",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)

    srcs: list[str] = []
    for n in nodes:
        if n.get("tag") == "img":
            srcs.append(n["attrs"]["src"])
        elif n.get("tag") == "figure":
            for ch in n.get("children", []):
                if isinstance(ch, dict) and ch.get("tag") == "img":
                    srcs.append(ch["attrs"]["src"])

    assert srcs
    assert all("catbox.moe" not in s for s in srcs)
    assert safe_gallery in srcs


@pytest.mark.asyncio
async def test_build_festival_page_content_uses_safe_fallback_cover(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    safe_cover = "https://example.com/fallback-cover.jpg"
    await main.set_setting_value(db, "festivals_index_cover", safe_cover)

    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            photo_url="https://files.catbox.moe/a.jpg",
            photo_urls=["https://files.catbox.moe/a.jpg", "https://files.catbox.moe/b.jpg"],
            description="desc",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)

    srcs: list[str] = []
    for n in nodes:
        if n.get("tag") == "img":
            srcs.append(n["attrs"]["src"])
        elif n.get("tag") == "figure":
            for ch in n.get("children", []):
                if isinstance(ch, dict) and ch.get("tag") == "img":
                    srcs.append(ch["attrs"]["src"])

    assert srcs and srcs[0] == safe_cover
    # When only legacy Catbox photos are available, we still keep them in the
    # gallery so operators can see illustrations on the festival page.
    assert len(srcs) >= 2
    assert any("catbox.moe" in s for s in srcs[1:])


@pytest.mark.asyncio
async def test_build_festival_page_content_hides_source_post_link(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            description="desc",
            source_post_url="https://t.me/channel/123",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "пост-источник" not in html
    assert "https://t.me/channel/123" not in html


@pytest.mark.asyncio
async def test_build_festival_page_content_hides_inferred_tg_channel(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            description="desc",
            tg_url="https://t.me/klassster",
            source_post_url="https://t.me/klassster/17700",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "Контакты фестиваля" not in html
    assert "https://t.me/klassster" not in html


@pytest.mark.asyncio
async def test_build_festival_page_content_hides_source_channel_even_with_other_links(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            description="desc",
            tg_url="https://t.me/klassster",
            website_url="https://example.com/fest",
            source_post_url="https://t.me/klassster/17700",
        )
        session.add(fest)
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "https://example.com/fest" in html
    assert "https://t.me/klassster" not in html


@pytest.mark.asyncio
async def test_build_festival_page_content_shows_confirmed_tg_channel(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="Линии тела",
            description="desc",
            tg_url="https://t.me/klassster",
            source_post_url="https://t.me/klassster/17700",
        )
        session.add(fest)
        src = (
            await session.execute(
                select(TelegramSource).where(TelegramSource.username == "klassster")
            )
        ).scalar_one_or_none()
        if src is None:
            src = TelegramSource(username="klassster", enabled=True)
            session.add(src)
        src.festival_source = True
        src.festival_series = "Линии тела"
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "Контакты фестиваля" in html
    assert "https://t.me/klassster" in html


@pytest.mark.asyncio
async def test_build_festival_page_content_shows_source_count(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    day = date.today().isoformat()
    async with db.get_session() as session:
        fest = Festival(
            name="Fest",
            description="desc",
            source_post_url="https://t.me/fest/100",
        )
        session.add(fest)
        ev1 = Event(
            title="A",
            description="d",
            date=day,
            time="",
            location_name="loc",
            source_text="src",
            festival="Fest",
            source_post_url="https://t.me/fest/100",
        )
        ev2 = Event(
            title="B",
            description="d",
            date=day,
            time="",
            location_name="loc",
            source_text="src",
            festival="Fest",
            source_post_url="https://vk.com/wall-1_2",
        )
        session.add(ev1)
        session.add(ev2)
        await session.flush()
        session.add(EventSource(event_id=ev1.id, source_type="tg", source_url="https://t.me/fest/100"))
        session.add(
            EventSource(
                event_id=ev1.id,
                source_type="web",
                source_url="https://example.com/news?id=42",
            )
        )
        session.add(
            EventSource(
                event_id=ev2.id,
                source_type="web",
                source_url="https://example.com/news?id=42",
            )
        )
        await session.commit()
        fid = fest.id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    _, nodes = await main.build_festival_page_content(db, fest)
    html = nodes_to_html(nodes)
    assert "📚 Источников: 3" in html


@pytest.mark.asyncio
async def test_handle_festival_edit_message_adds_image_link(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def nop(*a, **k):
        return None

    monkeypatch.setattr(main, "sync_festival_page", nop)
    monkeypatch.setattr(main, "sync_festivals_index_page", nop)

    async with db.get_session() as session:
        fest = Festival(name="Fest")
        session.add(fest)
        await session.commit()
        fid = fest.id

    user_id = 123
    main.festival_edit_sessions[user_id] = (fid, main.FESTIVAL_EDIT_FIELD_IMAGE)

    class DummyBot:
        def __init__(self):
            self.sent: list[dict[str, object]] = []

        async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
            self.sent.append(
                {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
            )

        async def download(self, *a, **k):  # pragma: no cover - guard
            raise AssertionError("download should not be called")

    bot = DummyBot()

    message = types.Message.model_validate(
        {
            "message_id": 1,
            "date": datetime.now(timezone.utc),
            "chat": {"id": 999, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Tester"},
            "text": "https://catbox.moe/new.jpg",
        }
    )

    await main.handle_festival_edit_message(message, db, bot)

    assert main.festival_edit_sessions[user_id] == (fid, None)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.photo_urls == ["https://catbox.moe/new.jpg"]
        assert fest.photo_url == "https://catbox.moe/new.jpg"

    main.festival_edit_sessions.pop(user_id, None)
