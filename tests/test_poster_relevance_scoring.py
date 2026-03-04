from __future__ import annotations


def test_score_eventposter_against_event_prefers_matching_title_date_time() -> None:
    import main

    s_good = main._score_eventposter_against_event(
        event_title="Disco Party",
        event_date="2026-02-22",
        event_time="19:00",
        ocr_title="DISCO PARTY",
        ocr_text="22.02 19:00",
    )
    s_bad = main._score_eventposter_against_event(
        event_title="Disco Party",
        event_date="2026-02-22",
        event_time="19:00",
        ocr_title="Valentine Party",
        ocr_text="15.02 16:00",
    )
    assert s_good > s_bad

