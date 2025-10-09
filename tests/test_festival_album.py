from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiogram import types

import main
from db import Database
from models import Festival


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
