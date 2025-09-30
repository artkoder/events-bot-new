import pytest
import httpx
from pathlib import Path

import main
from db import Database
from models import Event, Festival


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


@pytest.mark.asyncio
async def test_extract_cover_img_external(monkeypatch):
    main.telegraph_first_image.clear()

    async def fake_get(self, url):
        return FakeResponse(
            {
                "result": {
                    "content": [
                        {
                            "tag": "img",
                            "attrs": {"src": "https://files.catbox.moe/a.jpg"},
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    url = await main.extract_telegra_ph_cover_url("https://telegra.ph/test")
    assert url == "https://files.catbox.moe/a.jpg"


@pytest.mark.asyncio
async def test_extract_cover_link(monkeypatch):
    main.telegraph_first_image.clear()

    async def fake_get(self, url):
        return FakeResponse({"result": {"content": [{"tag": "a", "attrs": {"href": "https://telegra.ph/file/y.png"}}]}})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    url = await main.extract_telegra_ph_cover_url("https://telegra.ph/test2")
    assert url == "https://telegra.ph/file/y.png"


@pytest.mark.asyncio
async def test_try_set_fest_cover_from_program(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(name="Fest", program_url="https://telegra.ph/test")
        session.add(fest)
        await session.commit()
        fid = fest.id

    async def fake_extract(url, *, event_id=None):
        return "https://telegra.ph/file/cover.jpg"

    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        ok = await main.try_set_fest_cover_from_program(db, fest)
        assert ok

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.photo_url == "https://telegra.ph/file/cover.jpg"


@pytest.mark.asyncio
async def test_try_set_fest_cover_handles_none_photo_urls(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(name="Fest2", program_url="https://telegra.ph/none")
        fest.photo_urls = None
        session.add(fest)
        await session.commit()
        fid = fest.id

    async def fake_extract(url, *, event_id=None):
        return "https://telegra.ph/file/cover-none.jpg"

    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        ok = await main.try_set_fest_cover_from_program(db, fest)
        assert ok

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.photo_url == "https://telegra.ph/file/cover-none.jpg"
        assert fest.photo_urls == ["https://telegra.ph/file/cover-none.jpg"]


@pytest.mark.asyncio
async def test_try_set_fest_cover_uses_telegraph_url(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(name="Fest3", telegraph_url="https://telegra.ph/fallback")
        session.add(fest)
        await session.commit()
        fid = fest.id

    called_urls: list[str] = []

    async def fake_extract(url, *, event_id=None):
        called_urls.append(url)
        return "https://telegra.ph/file/cover-telegraph.jpg"

    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        ok = await main.try_set_fest_cover_from_program(db, fest)
        assert ok

    assert called_urls == ["https://telegra.ph/fallback"]

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.photo_url == "https://telegra.ph/file/cover-telegraph.jpg"
        assert fest.photo_urls == ["https://telegra.ph/file/cover-telegraph.jpg"]


@pytest.mark.asyncio
async def test_try_set_fest_cover_uses_event_media(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        fest = Festival(
            name="FestEvents",
            photo_urls=["https://example.com/existing.jpg"],
        )
        session.add(fest)
        await session.flush()

        event_with_photo = Event(
            title="Event 1",
            description="Desc",
            festival=fest.name,
            date="2024-01-01",
            time="10:00",
            location_name="Place",
            source_text="Source",
            photo_urls=["https://example.com/event-photo.jpg"],
        )
        event_with_telegra_ph = Event(
            title="Event 2",
            description="Desc",
            festival=fest.name,
            date="2024-01-02",
            time="11:00",
            location_name="Place",
            source_text="Source",
            telegraph_url="https://telegra.ph/event-cover",
        )
        session.add_all([event_with_photo, event_with_telegra_ph])
        await session.commit()
        fid = fest.id

    async def fake_extract(url, *, event_id=None):
        if url == "https://telegra.ph/event-cover":
            return "https://telegra.ph/file/event-cover.jpg"
        return None

    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        ok = await main.try_set_fest_cover_from_program(db, fest)
        assert ok

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.photo_urls == [
            "https://example.com/event-photo.jpg",
            "https://telegra.ph/file/event-cover.jpg",
            "https://example.com/existing.jpg",
        ]
        assert fest.photo_url == "https://example.com/event-photo.jpg"
