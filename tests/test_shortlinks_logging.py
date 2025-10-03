import logging

import pytest

from models import Event
from shortlinks import ensure_vk_short_ticket_link


def _make_event(**kwargs: object) -> Event:
    defaults = dict(
        id=1,
        title="Title",
        description="Description",
        date="2024-01-01",
        time="10:00",
        location_name="Location",
        source_text="Source",
    )
    defaults.update(kwargs)
    return Event(**defaults)


@pytest.mark.asyncio
async def test_ensure_vk_short_ticket_link_logs_reuse(caplog: pytest.LogCaptureFixture) -> None:
    event = _make_event(
        ticket_link="https://example.com",
        vk_ticket_short_url="https://vk.cc/reused",
        vk_ticket_short_key="reused",
    )

    caplog.set_level(logging.INFO)

    result = await ensure_vk_short_ticket_link(event, db=None)

    assert result == ("https://vk.cc/reused", "reused")
    assert "vk_shortlink_reused" in caplog.text
    assert "url=https://example.com" in caplog.text
    assert "short_url=https://vk.cc/reused" in caplog.text


@pytest.mark.asyncio
async def test_ensure_vk_short_ticket_link_logs_saved(caplog: pytest.LogCaptureFixture) -> None:
    async def fake_vk_api(method: str, params: dict[str, str], *_: object) -> dict[str, str]:
        assert method == "utils.getShortLink"
        assert params == {"url": "https://example.com"}
        return {"response": {"key": "abcd", "short_url": "https://vk.cc/abcd"}}

    event = _make_event(ticket_link="https://example.com", vk_ticket_short_url=None, vk_ticket_short_key=None)

    caplog.set_level(logging.INFO)

    result = await ensure_vk_short_ticket_link(event, db=None, vk_api_fn=fake_vk_api)

    assert result == ("https://vk.cc/abcd", "abcd")
    assert "vk_shortlink_saved" in caplog.text
    assert "url=https://example.com" in caplog.text
    assert "short_url=https://vk.cc/abcd" in caplog.text


@pytest.mark.asyncio
async def test_ensure_vk_short_ticket_link_logs_fallback(caplog: pytest.LogCaptureFixture) -> None:
    event = _make_event(ticket_link="  ")

    caplog.set_level(logging.INFO)

    result = await ensure_vk_short_ticket_link(event, db=None)

    assert result is None
    assert "vk_shortlink_fallback" in caplog.text
    assert "reason=empty_ticket_link" in caplog.text
