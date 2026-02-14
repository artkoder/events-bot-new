from datetime import date

from smart_event_update import EventCandidate
from source_parsing.telegram.handlers import _should_skip_past_event_candidate


def _candidate(*, event_type: str | None, start: str | None, end: str | None) -> EventCandidate:
    return EventCandidate(
        source_type="telegram",
        source_url="https://t.me/example/1",
        source_text="stub",
        title="stub",
        event_type=event_type,
        date=start,
        end_date=end,
    )


def test_regular_past_event_is_skipped() -> None:
    c = _candidate(event_type="концерт", start="2026-02-03", end=None)
    assert _should_skip_past_event_candidate(c, today=date(2026, 2, 11)) is True


def test_regular_future_event_is_not_skipped() -> None:
    c = _candidate(event_type="концерт", start="2026-02-12", end=None)
    assert _should_skip_past_event_candidate(c, today=date(2026, 2, 11)) is False


def test_exhibition_with_future_end_date_is_not_skipped() -> None:
    c = _candidate(event_type="exhibition", start="2026-02-03", end="2026-04-03")
    assert _should_skip_past_event_candidate(c, today=date(2026, 2, 11)) is False


def test_exhibition_without_end_date_is_not_skipped() -> None:
    c = _candidate(event_type="выставка", start="2026-02-03", end=None)
    assert _should_skip_past_event_candidate(c, today=date(2026, 2, 11)) is False


def test_exhibition_with_past_end_date_is_skipped() -> None:
    c = _candidate(event_type="выставка", start="2026-01-01", end="2026-02-01")
    assert _should_skip_past_event_candidate(c, today=date(2026, 2, 11)) is True
