import asyncio

import pytest

import main


class DummyResp:
    def __init__(self, status: int, text: str):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def post(self, url, data):
        return next(self._responses)


@pytest.mark.asyncio
async def test_upload_images_catbox_ok(monkeypatch):
    main.CATBOX_ENABLED = True
    resp = DummyResp(200, "http://cat/1.png")
    monkeypatch.setattr(main, "get_http_session", lambda: DummySession([resp]))
    monkeypatch.setattr(main, "telegraph_upload", lambda d, n: None)
    urls, tg_urls, msg = await main.upload_images([(b"1", "a.png")])
    assert urls == ["http://cat/1.png"]
    assert tg_urls == []
    assert "ok" in msg


@pytest.mark.asyncio
async def test_upload_images_fallback(monkeypatch):
    main.CATBOX_ENABLED = True
    resp = DummyResp(412, "pause")
    monkeypatch.setattr(main, "get_http_session", lambda: DummySession([resp, resp, resp]))
    monkeypatch.setattr(main, "telegraph_upload", lambda d, n: "https://telegra.ph/file/tg.png")
    monkeypatch.setattr(asyncio, "sleep", lambda s: None)
    urls, tg_urls, msg = await main.upload_images([(b"1", "a.png")])
    assert urls == []
    assert tg_urls == ["https://telegra.ph/file/tg.png"]
    assert "tg ok" in msg


@pytest.mark.asyncio
async def test_upload_images_both_fail(monkeypatch):
    main.CATBOX_ENABLED = True
    resp = DummyResp(500, "err")
    monkeypatch.setattr(main, "get_http_session", lambda: DummySession([resp, resp, resp]))
    monkeypatch.setattr(main, "telegraph_upload", lambda d, n: None)
    monkeypatch.setattr(asyncio, "sleep", lambda s: None)
    urls, tg_urls, msg = await main.upload_images([(b"1", "a.png")])
    assert urls == []
    assert tg_urls == []
    assert "both failed" in msg

