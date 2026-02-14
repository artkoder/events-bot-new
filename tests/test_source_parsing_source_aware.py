from __future__ import annotations

import pytest

from source_parsing.handlers import process_source_events
from source_parsing.parser import TheatreEvent


def _make_event() -> TheatreEvent:
    return TheatreEvent(
        title="Фигаро",
        date_raw="27 февраля 18:00",
        parsed_date="2099-02-27",
        parsed_time="18:00",
        ticket_status="available",
        url="https://dramteatr39.ru/spektakli/figaro",
        location="Драматический театр",
        source_type="dramteatr",
        photos=[],
        description="Тестовое описание",
    )


@pytest.mark.asyncio
async def test_existing_without_parser_source_goes_through_smart_update(monkeypatch):
    event = _make_event()

    async def _find_existing(*_args, **_kwargs):
        return 100, False

    async def _no_parser_source(*_args, **_kwargs):
        return False

    async def _smart_merge(*_args, **_kwargs):
        return 100, False, "merged"

    async def _fail_ticket_update(*_args, **_kwargs):
        raise AssertionError("ticket-only path must not run when parser source is missing")

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("source_parsing.handlers.find_existing_event", _find_existing)
    monkeypatch.setattr("source_parsing.handlers.event_has_parser_source", _no_parser_source)
    monkeypatch.setattr("source_parsing.handlers.add_new_event_via_queue", _smart_merge)
    monkeypatch.setattr("source_parsing.handlers.update_event_ticket_status", _fail_ticket_update)
    monkeypatch.setattr("source_parsing.handlers.build_updated_event_info", _noop)
    monkeypatch.setattr("source_parsing.handlers.EVENT_ADD_DELAY_SECONDS", 0)

    stats, _ = await process_source_events(
        db=object(),  # not used by mocked branches
        bot=None,
        events=[event],
        source="dramteatr",
        start_index=0,
        total_count=1,
    )

    assert stats.new_added == 0
    assert stats.ticket_updated == 1
    assert stats.skipped == 0
    assert stats.failed == 0


@pytest.mark.asyncio
async def test_existing_with_parser_source_uses_ticket_sync_only(monkeypatch):
    event = _make_event()

    async def _find_existing(*_args, **_kwargs):
        return 101, False

    async def _has_parser_source(*_args, **_kwargs):
        return True

    async def _ticket_updated(*_args, **_kwargs):
        return True

    async def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("smart update path must not run when parser source already exists")

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr("source_parsing.handlers.find_existing_event", _find_existing)
    monkeypatch.setattr("source_parsing.handlers.event_has_parser_source", _has_parser_source)
    monkeypatch.setattr("source_parsing.handlers.update_event_ticket_status", _ticket_updated)
    monkeypatch.setattr("source_parsing.handlers.schedule_existing_event_update", _noop)
    monkeypatch.setattr("source_parsing.handlers.update_linked_events", _noop)
    monkeypatch.setattr("source_parsing.handlers.add_new_event_via_queue", _raise_if_called)

    stats, _ = await process_source_events(
        db=object(),  # not used by mocked branches
        bot=None,
        events=[event],
        source="dramteatr",
        start_index=0,
        total_count=1,
    )

    assert stats.new_added == 0
    assert stats.ticket_updated == 1
    assert stats.skipped == 0
    assert stats.failed == 0


@pytest.mark.asyncio
async def test_smart_update_skipped_status_is_counted_as_skipped(monkeypatch):
    event = _make_event()

    async def _find_existing(*_args, **_kwargs):
        return 102, False

    async def _no_parser_source(*_args, **_kwargs):
        return False

    async def _smart_skipped(*_args, **_kwargs):
        return None, False, "skipped_nochange"

    monkeypatch.setattr("source_parsing.handlers.find_existing_event", _find_existing)
    monkeypatch.setattr("source_parsing.handlers.event_has_parser_source", _no_parser_source)
    monkeypatch.setattr("source_parsing.handlers.add_new_event_via_queue", _smart_skipped)

    stats, _ = await process_source_events(
        db=object(),  # not used by mocked branches
        bot=None,
        events=[event],
        source="dramteatr",
        start_index=0,
        total_count=1,
    )

    assert stats.new_added == 0
    assert stats.ticket_updated == 0
    assert stats.skipped == 1
    assert stats.failed == 0
