import logging
import pytest

import main


class FakeTG:
    def __init__(self, html):
        self.html = html

    def get_page(self, path, return_html=False):
        return {"content": self.html}


async def fake_telegraph_call(func, *args, **kwargs):
    return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_check_month_page_markers_missing(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    tg = FakeTG("<p>no markers</p>")
    await main.check_month_page_markers(tg, "path")
    assert "month_rebuild_markers_missing" in caplog.text


@pytest.mark.asyncio
async def test_check_month_page_markers_present(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(main, "telegraph_call", fake_telegraph_call)
    tg = FakeTG("<!--DAY:2025-08-24 START-->")
    await main.check_month_page_markers(tg, "path")
    assert "month_rebuild_markers_present" in caplog.text
