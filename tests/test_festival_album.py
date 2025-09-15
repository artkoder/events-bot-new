import pytest
from pathlib import Path

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
    imgs: list[str] = []
    for n in nodes:
        if n.get("tag") == "img":
            imgs.append(n["attrs"]["src"])
        elif n.get("tag") == "figure":
            for child in n.get("children", []):
                if child.get("tag") == "img":
                    imgs.append(child["attrs"]["src"])
                    break
    assert imgs == [urls[1], urls[0], urls[2]]
