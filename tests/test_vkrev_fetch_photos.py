import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

import main

main.VK_TOKEN_AFISHA = "ga"


@pytest.mark.asyncio
async def test_copy_history_photo(monkeypatch):
    post_id = 1

    async def fake_vk_api(method, **params):
        assert method == "wall.getById"
        assert params.get("posts") == f"-1_{post_id}"
        return [
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
                        ],
                    }
                ],
                "attachments": [],
            }
        ]

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, post_id, None, None)
    assert photos == ["http://p"]


@pytest.mark.asyncio
async def test_link_preview(monkeypatch):
    post_id = 2

    async def fake_vk_api(method, **params):
        assert params.get("posts") == f"-1_{post_id}"
        return [
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

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, post_id, None, None)
    assert photos == ["http://l"]


@pytest.mark.asyncio
async def test_video_preview(monkeypatch):
    calls = []

    post_id = 3

    async def fake_vk_api(method, **params):
        calls.append((method, params.get("posts")))
        return [
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

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, post_id, None, None)
    assert photos == ["http://v"]
    assert calls == [("wall.getById", "-1_3")]


@pytest.mark.asyncio
async def test_doc_preview(monkeypatch):
    post_id = 4

    async def fake_vk_api(method, **params):
        return [
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

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, post_id, None, None)
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

    post_id = 5

    async def fake_vk_api(method, **params):
        return [
            {
                "copy_history": [{"attachments": copy_atts}],
                "attachments": atts,
            }
        ]

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    photos = await main._vkrev_fetch_photos(1, post_id, None, None)
    assert len(photos) == 10
    assert len(set(photos)) == 10
    assert photos[:6] == urls[:6]
