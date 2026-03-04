from source_parsing.telegram.handlers import (
    _build_candidate,
    _filter_posters_for_event,
    _filter_schedule_source_text,
    _infer_time_from_event_text,
)
from models import TelegramSource
from smart_event_update import PosterCandidate


def test_filter_schedule_source_text_by_date_token():
    raw = (
        "07.02 | Мертвые души\n"
        "неделя в театре\n"
        "03.02 | Нюрнберг\n"
        "04.02 | Мысли мудрых людей на каждый день\n"
        "08.02 | Три супруги-совершенства\n"
    )
    out = _filter_schedule_source_text(raw, event_date="2026-02-07", event_title="Мёртвые души")
    assert "Мертвые души" in out
    assert "Мысли мудрых" not in out
    assert "Три супруги" not in out


def test_filter_schedule_source_text_single_event_passthrough():
    raw = "07.02 | Мертвые души"
    out = _filter_schedule_source_text(raw, event_date="2026-02-07", event_title="Мёртвые души")
    assert out == raw


def test_filter_schedule_source_text_keeps_figaro_fact_line():
    raw = (
        "06.02 | Пижама на шестерых\n"
        "Мечта о тайном от жены свидании с любовницей оборачивается вечеринкой с участием всех жен и любовниц.\n\n"
        "08.02 | Три супруги-совершенства\n"
        "Мелодрама с испанскими мотивами о любви, изменах и разоблачениях.\n\n"
        "12.02 | Фигаро\n"
        "Нескучная французская классика. Настоящий театральный хит, проверенный временем."
    )
    out = _filter_schedule_source_text(raw, event_date="2026-02-12", event_title="Фигаро")
    assert "12.02 | Фигаро" in out
    assert "Нескучная французская классика" in out
    assert "Пижама на шестерых" not in out
    assert "Три супруги-совершенства" not in out


def test_filter_posters_for_event_drops_foreign_posters():
    posters = [
        PosterCandidate(ocr_text="НЮРНБЕРГ\n3.02 19:00"),
        PosterCandidate(ocr_text="МЁРТВЫЕ ДУШИ\n7.02 18:00"),
        PosterCandidate(ocr_text="ПИЖАМА НА ШЕСТЕРЫХ\n6.02 19:00"),
        PosterCandidate(ocr_text="НЕДЕЛЯ В ТЕАТРЕ\nНЮРНБЕРГ 3.02\nМЁРТВЫЕ ДУШИ 7.02\nТРИ СУПРУГИ СОВЕРШЕНСТВА 8.02"),
    ]
    kept = _filter_posters_for_event(
        posters,
        event_title="Мёртвые души",
        event_date="2026-02-07",
        event_time="18:00",
    )
    assert len(kept) == 1
    assert "МЁРТВЫЕ ДУШИ" in (kept[0].ocr_text or "")


def test_filter_posters_for_event_returns_empty_when_no_specific_poster():
    posters = [
        PosterCandidate(ocr_text="МЁРТВЫЕ ДУШИ\n7.02 18:00"),
        PosterCandidate(ocr_text="ПИЖАМА НА ШЕСТЕРЫХ\n6.02 19:00"),
        PosterCandidate(ocr_text="НЕДЕЛЯ В ТЕАТРЕ\n... 8.02 ТРИ СУПРУГИ СОВЕРШЕНСТВА ..."),
    ]
    kept = _filter_posters_for_event(
        posters,
        event_title="Три супруги-совершенства",
        event_date="2026-02-08",
        event_time="18:00",
    )
    assert kept == []


def test_build_candidate_keeps_assigned_event_poster_when_time_missing():
    source = TelegramSource(
        id=1,
        username="dramteatr39",
        enabled=True,
        default_location="Драматический театр",
    )
    message = {
        "source_username": "dramteatr39",
        "message_id": 3782,
        "source_link": "https://t.me/dramteatr39/3782",
        "text": (
            "12.02 | Фигаро\n"
            "Нескучная французская классика. Настоящий театральный хит, проверенный временем."
        ),
        "posters": [
            {
                "sha256": "generic-1",
                "catbox_url": "https://files.catbox.moe/generic1.jpg",
                "ocr_text": "НЕДЕЛЯ В ТЕАТРЕ\n06.02\n08.02\n12.02",
            }
        ],
    }
    event_data = {
        "title": "Фигаро",
        "date": "2026-02-12",
        "time": "",
        "location_name": "Драматический театр",
        "source_text": "12.02 | Фигаро",
        "posters": [
            {
                "sha256": "figaro-1",
                "catbox_url": "https://files.catbox.moe/figaro.jpg",
                "ocr_text": "12.02 ФИГАРО",
            }
        ],
    }

    candidate = _build_candidate(source, message, event_data)

    assert len(candidate.posters) == 1
    assert candidate.posters[0].sha256 == "figaro-1"


def test_build_candidate_normalizes_english_exhibition_type():
    source = TelegramSource(
        id=1,
        username="tretyakovka_kaliningrad",
        enabled=True,
        default_location="Третьяковская галерея",
    )
    message = {
        "source_username": "tretyakovka_kaliningrad",
        "message_id": 2391,
        "source_link": "https://t.me/tretyakovka_kaliningrad/2391",
        "text": "Остается пара месяцев до окончания работы выставки «Пять веков русского искусства».",
        "posters": [],
    }
    event_data = {
        "title": "Пять веков русского искусства",
        "date": "2026-02-03",
        "end_date": "2026-04-03",
        "event_type": "exhibition",
        "location_name": "Третьяковская галерея",
        "source_text": "Остается пара месяцев до окончания работы выставки «Пять веков русского искусства».",
    }

    candidate = _build_candidate(source, message, event_data)
    assert candidate.event_type == "выставка"


def test_infer_time_from_event_text_uses_unique_time_for_event_date():
    text = (
        "12.02 | Фигаро\n"
        "\"ФИГАРО\"\n"
        "НЕСКУЧНАЯ ФРАНЦУЗСКАЯ КЛАССИКА\n"
        "12.02 19:00"
    )
    assert _infer_time_from_event_text(text, event_date="2026-02-12") == "19:00"


def test_build_candidate_infers_missing_time_from_event_source_text():
    source = TelegramSource(
        id=1,
        username="dramteatr39",
        enabled=True,
        default_location="Драматический театр",
    )
    message = {
        "source_username": "dramteatr39",
        "message_id": 3821,
        "source_link": "https://t.me/dramteatr39/3821",
        "text": (
            "неделя в театре\n\n"
            "11.02 | Коралина\n"
            "12.02 | Фигаро\n"
            "13.02 | Дикарь"
        ),
        "posters": [],
    }
    event_data = {
        "title": "Фигаро",
        "date": "2026-02-12",
        "time": "",
        "location_name": "Драматический театр",
        "source_text": (
            "12.02 | Фигаро\n\n"
            "\"ФИГАРО\"\n"
            "НЕСКУЧНАЯ ФРАНЦУЗСКАЯ КЛАССИКА\n"
            "12.02 19:00"
        ),
    }
    candidate = _build_candidate(source, message, event_data)
    assert candidate.time == "19:00"
