from datetime import datetime as real_datetime, timezone

import pytest

import main
import vk_intake
from vk_intake import match_keywords, detect_date, extract_event_ts_hint


def test_match_keywords_variants():
    ok, kws = match_keywords("#спектакль сегодня 20:00")
    assert ok
    assert any("спект" in k for k in kws)


def test_match_keywords_music_invite_pushkin_card():
    text = "Приглашаем на музыкальный вечер 25 августа, доступно по Пушкинской карте"
    ok, kws = match_keywords(text)
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("приглашаем") for k in normalized)
    assert any(k.startswith("музык") for k in normalized)
    assert any("пушкинск" in k and "карт" in k for k in normalized)


def test_match_keywords_lead_host_variants():
    ok, kws = match_keywords("Ведущая расскажет о программе")
    assert ok
    assert any("ведущ" in k for k in kws)

    ok_hash, kws_hash = match_keywords("#Ведущий поделится опытом")
    assert ok_hash
    assert any("ведущ" in k for k in kws_hash)


def test_match_keywords_cost_variants():
    ok_free, kws_free = match_keywords("Вход свободный, приходите 12 мая")
    assert ok_free
    assert any("вход свободн" in k for k in kws_free)

    ok_paid, kws_paid = match_keywords("Билеты по 500 руб. и регистрация обязательна")
    assert ok_paid
    assert any("500 руб" in k or "руб." in k for k in kws_paid)

    ok_symbol, kws_symbol = match_keywords("Стоимость участия — 1 200₽")
    assert ok_symbol
    assert any("₽" in k for k in kws_symbol)


def test_match_keywords_fest_and_karaoke():
    ok, kws = match_keywords("КАШТАН FEST 3 октября 17:00 — жаркое караоке")
    assert ok
    normalized = {k.lower() for k in kws}
    assert "fest" in normalized
    assert "караоке" in normalized


def test_match_keywords_prazdnik():
    ok, kws = match_keywords("1 октября — праздник двора, вход свободный")
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("праздник") for k in normalized)


def test_match_keywords_events_digest():
    ok, kws = match_keywords("Дайджест событий выходных 7–8 сентября")
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("событ") for k in normalized)


def test_match_keywords_digest_word():
    ok, kws = match_keywords("Культурный дайджест: лучшие события 10 сентября")
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("дайджест") for k in normalized)


def test_match_keywords_tribute():
    ok, kws = match_keywords("Сегодня трибьют группы Queen 1 сентября 20:00")
    assert ok
    normalized = {k.lower() for k in kws}
    assert "трибьют" in normalized


def test_match_keywords_tribute_with_punctuation():
    ok, kws = match_keywords("17 октября — трибьют группы ... 20:00")
    assert ok
    normalized = {k.lower() for k in kws}
    assert "трибьют" in normalized


def test_match_keywords_concert_phrases():
    text = (
        "Все хиты группы Лучшие песни в исполнении группы Мечта "
        "два часа живого звука 3 сентября"
    )
    ok, kws = match_keywords(text)
    assert ok
    normalized = {k.lower() for k in kws}
    assert any("хит" in k for k in normalized)
    assert any(k.startswith("групп") for k in normalized)
    assert any("исполнен" in k for k in normalized)
    assert any("жив" in k and "звук" in k for k in normalized)


def test_match_keywords_hits_with_group_context():
    ok, kws = match_keywords("Все хиты группы «Четыре» 21 ноября 19:00")
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("хит") for k in normalized)
    assert any(k.startswith("групп") and "«" in k for k in normalized)


def test_match_keywords_live_sound_and_performance():
    ok, kws = match_keywords(
        "Два часа живого звука в живом исполнении и особое выступление 17 октября"
    )
    assert ok
    normalized = {k.lower() for k in kws}
    assert any("жив" in k and "звук" in k for k in normalized)
    assert any("жив" in k and "исполнен" in k for k in normalized)
    assert any(k.startswith("выступлен") for k in normalized)


def test_match_keywords_poetry_songs_play():
    ok, kws = match_keywords("стихи по кругу, сыграем песни 5 октября")
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("стих") for k in normalized)
    assert any(k.startswith("сыгра") for k in normalized)
    assert any(k.startswith("песн") for k in normalized)


def test_match_keywords_piano_works_composers():
    text = "фортепианные дуэты, в программе произведения композиторов 24 августа"
    ok, kws = match_keywords(text)
    assert ok
    normalized = {k.lower() for k in kws}
    assert any(k.startswith("фортепиан") for k in normalized)
    assert any("в программе" in k and "произведен" in k for k in normalized)
    assert any(k.startswith("композитор") for k in normalized)


def test_detect_date_and_extract():
    text = "Мастер-классы 14–15.09, регистрация по ссылке"
    assert detect_date(text)

    publish_dt = real_datetime(2024, 8, 1, tzinfo=main.LOCAL_TZ)
    ts = extract_event_ts_hint(text, publish_ts=publish_dt)
    assert ts is not None
    dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (dt.year, dt.month, dt.day) == (2024, 9, 15)


def test_extract_event_ts_hint_recent_past():
    publish_dt = real_datetime(2024, 10, 1, tzinfo=main.LOCAL_TZ)
    past_text = "7 сентября прошла лекция"
    assert extract_event_ts_hint(past_text, publish_ts=publish_dt) is None

    future_text = "7 января состоится концерт"
    ts = extract_event_ts_hint(future_text, publish_ts=publish_dt)
    assert ts is not None
    future_dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (future_dt.year, future_dt.month, future_dt.day) == (2025, 1, 7)


def test_extract_event_ts_hint_explicit_year_past():
    publish_dt = real_datetime(2026, 1, 1, tzinfo=main.LOCAL_TZ)
    text = "Концерт состоится 17 сентября 2025 года"
    assert extract_event_ts_hint(text, publish_ts=publish_dt) is None


def test_extract_event_ts_hint_ignores_phone_number_segments():
    publish_dt = real_datetime(2024, 4, 1, tzinfo=main.LOCAL_TZ)
    text = "Встречаемся в пт, звоните 474-30-04"
    ts = extract_event_ts_hint(text, publish_ts=publish_dt)
    assert ts is not None
    dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (dt.year, dt.month, dt.day) == (2024, 4, 5)


def test_extract_event_ts_hint_phone_like_sequence_only():
    publish_dt = real_datetime(2024, 4, 1, tzinfo=main.LOCAL_TZ)
    text = "Запись по телефону 8 (4012) 27-01-26"
    assert extract_event_ts_hint(text, publish_ts=publish_dt) is None


def test_extract_event_ts_hint_weekday_uses_publish_week(monkeypatch):
    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            tzinfo = tz or timezone.utc
            return real_datetime(2024, 5, 13, tzinfo=tzinfo)

    monkeypatch.setattr("vk_intake.datetime", FixedDatetime)

    publish_dt = FixedDatetime(2024, 5, 6, tzinfo=main.LOCAL_TZ)
    ts = extract_event_ts_hint("В ср встречаемся", publish_ts=publish_dt)
    assert ts is not None
    dt = real_datetime.fromtimestamp(ts, tz=main.LOCAL_TZ)
    assert (dt.year, dt.month, dt.day) == (2024, 5, 8)


@pytest.mark.asyncio
async def test_build_drafts_library_defaults_to_free(monkeypatch):
    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Лекция в библиотеке",
                "location_name": "Центральная библиотека",
                "location_address": "ул. Ленина, 10",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse, raising=False)

    drafts = await vk_intake.build_event_drafts_from_vk(
        "Встреча читателей в уютной библиотеке"
    )

    assert drafts and drafts[0].is_free is True


@pytest.mark.asyncio
async def test_build_drafts_library_respects_paid_keywords(monkeypatch):
    async def fake_parse(*args, **kwargs):
        return [
            {
                "title": "Лекция в библиотеке",
                "location_name": "Центральная библиотека",
                "location_address": "ул. Ленина, 10",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse, raising=False)

    drafts = await vk_intake.build_event_drafts_from_vk(
        "Встреча читателей в библиотеке, вход 300 руб."
    )

    assert drafts and drafts[0].is_free is False
