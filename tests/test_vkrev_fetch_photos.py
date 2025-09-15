import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

import main

main.VK_TOKEN_AFISHA = "ga"


@pytest.mark.asyncio
async def test_copy_history_photo(monkeypatch):
    async def fake_vk_api(method, params, db, bot, **kwargs):
        assert method == "wall.getById"
        assert kwargs.get("token_kind") == "group"
        assert kwargs.get("token") == main.VK_TOKEN_AFISHA
        assert kwargs.get("skip_captcha") is True
        return {
            "response": [
                {
                    "copy_history": [
                        {
                            "attachments": [
                                {
                                    "type": "photo",
                                    "photo": {
                                        "sizes": [
                                            {"width": 100, "height": 100, "url": "http://p"}
                                        ]
                                    },
                                }
                            ]
                        }
                    ],
                    "attachments": [],
                }
            ]
        }

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, 1, None, None)
    assert photos == ["http://p"]


@pytest.mark.asyncio
async def test_link_preview(monkeypatch):
    async def fake_vk_api(method, params, db, bot, **kwargs):
        return {
            "response": [
                {
                    "attachments": [
                        {
                            "type": "link",
                            "link": {
                                "photo": {
                                    "sizes": [
                                        {
                                            "width": 10,
                                            "height": 10,
                                            "url": "http://l",
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, 2, None, None)
    assert photos == ["http://l"]


@pytest.mark.asyncio
async def test_video_preview(monkeypatch):
    calls = []

    async def fake_vk_api(method, params, db, bot, **kwargs):
        calls.append(method)
        return {
            "response": [
                {
                    "attachments": [
                        {
                            "type": "video",
                            "video": {
                                "image": [
                                    {"width": 10, "height": 10, "url": "http://v"}
                                ]
                            },
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, 3, None, None)
    assert photos == ["http://v"]
    assert calls == ["wall.getById"]


@pytest.mark.asyncio
async def test_doc_preview(monkeypatch):
    async def fake_vk_api(method, params, db, bot, **kwargs):
        return {
            "response": [
                {
                    "attachments": [
                        {
                            "type": "doc",
                            "doc": {
                                "preview": {
                                    "photo": {
                                        "sizes": [
                                            {
                                                "width": 10,
                                                "height": 10,
                                                "url": "http://d",
                                            }
                                        ]
                                    }
                                }
                            },
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, 4, None, None)
    assert photos == ["http://d"]


@pytest.mark.asyncio
async def test_dedup_and_limit(monkeypatch):
    urls = [f"http://{i}" for i in range(12)]

    def make_photo(url):
        return {
            "type": "photo",
            "photo": {"sizes": [{"width": 1, "height": 1, "url": url}]},
        }

    copy_atts = [make_photo(u) for u in urls[:6]]
    atts = [make_photo(u) for u in urls[5:]]  # overlap at url[5]

    async def fake_vk_api(method, params, db, bot, **kwargs):
        return {
            "response": [
                {
                    "copy_history": [{"attachments": copy_atts}],
                    "attachments": atts,
                }
            ]
        }

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, 5, None, None)
    assert len(photos) == 10
    assert len(set(photos)) == 10
    assert photos[:6] == urls[:6]
