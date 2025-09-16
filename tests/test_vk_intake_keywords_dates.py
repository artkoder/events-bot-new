from datetime import datetime as real_datetime, timezone

import main
from vk_intake import match_keywords, detect_date, extract_event_ts_hint


def test_match_keywords_variants():
    ok, kws = match_keywords("#спектакль сегодня 20:00")
    assert ok
    assert any("спект" in k for k in kws)


def test_detect_date_and_extract(monkeypatch):
    text = "Мастер-классы 14–15.09, регистрация по ссылке"
    assert detect_date(text)

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            tzinfo = tz or timezone.utc
            return real_datetime(2024, 8, 1, tzinfo=tzinfo)

    monkeypatch.setattr("vk_intake.datetime", FixedDatetime)
    ts = extract_event_ts_hint(text)
    assert ts is not None
    dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (dt.year, dt.month, dt.day) == (2024, 9, 15)


def test_extract_event_ts_hint_recent_past(monkeypatch):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            tzinfo = tz or timezone.utc
            return real_datetime(2024, 10, 1, tzinfo=tzinfo)

    monkeypatch.setattr("vk_intake.datetime", FixedDatetime)

    past_text = "7 сентября прошла лекция"
    assert extract_event_ts_hint(past_text) is None

    future_text = "7 января состоится концерт"
    ts = extract_event_ts_hint(future_text)
    assert ts is not None
    future_dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (future_dt.year, future_dt.month, future_dt.day) == (2025, 1, 7)
