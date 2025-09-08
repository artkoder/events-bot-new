import pytest
import httpx
from pathlib import Path

import main
from db import Database
from models import Festival


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
