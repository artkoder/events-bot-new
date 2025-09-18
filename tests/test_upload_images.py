import asyncio
import logging

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

    def post(self, url, data):
        return next(self._responses)


@pytest.mark.asyncio
async def test_upload_images_catbox_ok(monkeypatch):
    main.CATBOX_ENABLED = True
    resp = DummyResp(200, "http://cat/1.png")
    monkeypatch.setattr(main, "get_http_session", lambda: DummySession([resp]))
    monkeypatch.setattr(main, "detect_image_type", lambda *a, **k: "jpeg")
    urls, msg = await main.upload_images([(b"1", "a.png")])
    assert urls == ["http://cat/1.png"]
    assert "ok" in msg


@pytest.mark.asyncio
async def test_upload_images_fail(monkeypatch):
    main.CATBOX_ENABLED = True
    resp = DummyResp(500, "err")
    monkeypatch.setattr(main, "get_http_session", lambda: DummySession([resp, resp, resp]))
    async def dummy_sleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", dummy_sleep)
    monkeypatch.setattr(main, "detect_image_type", lambda *a, **k: "jpeg")
    urls, msg = await main.upload_images([(b"1", "a.png")])
    assert urls == []
    assert "failed" in msg


@pytest.mark.asyncio
async def test_upload_images_catbox_disabled(monkeypatch, caplog):
    main.CATBOX_ENABLED = False
    caplog.set_level(logging.INFO)
    urls, msg = await main.upload_images([(b"1", "a.png")], event_hint="test")
    assert urls == []
    assert msg == "disabled"
    assert any(
        "CATBOX disabled catbox_enabled=False force=False images=1 event_hint=test"
        in record.message
        for record in caplog.records
    )

