import time
from vk_intake import match_keywords, detect_date, extract_event_ts_hint


def test_match_keywords_variants():
    ok, kws = match_keywords("#спектакль сегодня 20:00")
    assert ok
    assert any("спект" in k for k in kws)


def test_detect_date_and_extract():
    text = "Мастер-классы 14–15.09, регистрация по ссылке"
    assert detect_date(text)
    ts = extract_event_ts_hint(text)
    assert ts is not None and ts > int(time.time()) - 10
