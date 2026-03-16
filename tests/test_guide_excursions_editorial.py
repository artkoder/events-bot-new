from __future__ import annotations

import pytest

from guide_excursions import digest_writer
from guide_excursions import editorial


def test_apply_editorial_fallback_populates_line_fields_and_neutralizes_relative_text():
    item, reason = editorial.apply_editorial_fallback(
        {
            "canonical_title": "восток-2",
            "digest_blurb": "Завтра в 09:00 отправляемся в путешествие по восточному маршруту.",
            "summary_one_liner": "Завтра в 09:00 отправляемся в путешествие по восточному маршруту.",
            "time": "09:00",
            "duration_text": "10 часов",
            "route_summary": "ростверк ДС Центральной площади — восточный маршрут — возвращение вечером",
            "guide_names": ["Андрей Левченков"],
            "audience_fit": ["любителям истории", "тем, кто любит длинные выезды", "для группы"],
            "group_format": "для группы",
            "meeting_point": "ростверк ДС Центральной площади",
            "price_text": "2200 руб.",
            "source_about_text": "Запись: @alev701_admin",
            "booking_text": "",
            "booking_url": "",
            "dedup_source_text": "15 марта приглашаем в путешествие на Восток-2, запись открыта.",
        },
        date_label="Вс, 15 марта, 09:00",
    )

    assert reason == "fallback"
    assert item is not None
    assert item["canonical_title"] == "Путешествие на восток-2"
    assert "Завтра" not in item["digest_blurb"]
    assert "Вс, 15 марта, 09:00" in item["digest_blurb"]
    assert item["guide_line"] == "Андрей Левченков"
    assert item["schedule_line"] == "Вс, 15 марта, 09:00"
    assert item["audience_line"] == "любителям истории, тем, кто любит длинные выезды"
    assert item["group_format_line"] == "для группы"
    assert item["route_line"].startswith("ростверк ДС Центральной площади")
    assert item["duration_line"] == "10 часов"
    assert item["meeting_point_line"] == "ростверк ДС Центральной площади"
    assert item["price_line"] == "2200 руб."
    assert item["booking_text"] == "@alev701_admin"
    assert item["booking_url"] == "https://t.me/alev701_admin"
    assert item["booking_line"] == "@alev701_admin"


def test_apply_editorial_fallback_normalizes_generic_call_to_profile_phone():
    item, reason = editorial.apply_editorial_fallback(
        {
            "canonical_title": "Третьяковская галерея и история Калининграда",
            "digest_blurb": "Экскурсия по выставке и городу.",
            "summary_one_liner": "Экскурсия по выставке и городу.",
            "source_about_text": "+7 962 255-54-91 - Татьяна http://t.me/tatamartynyuk",
            "source_about_links": ["http://t.me/tatamartynyuk"],
            "booking_text": "Звоните",
            "booking_url": "",
            "dedup_source_text": "26 марта приглашаем школьные группы на экскурсию, запись открыта.",
        },
        date_label="Чт, 26 марта, 10:30",
    )

    assert reason == "fallback"
    assert item is not None
    assert item["booking_text"] == "+7 962 255-54-91"
    assert item["booking_url"] == "tel:+79622555491"
    assert item["booking_line"] == "+7 962 255-54-91"


def test_apply_editorial_fallback_prefers_mobile_phone_over_telegram_when_both_present():
    item, reason = editorial.apply_editorial_fallback(
        {
            "canonical_title": "Пешеходная прогулка по Светлогорску",
            "digest_blurb": "Прогулка по курортному городу.",
            "summary_one_liner": "Прогулка по курортному городу.",
            "source_about_text": "+7 921 710-11-61, +7 4012 521-161, @guide_contact",
            "source_about_links": ["https://t.me/guide_contact"],
            "booking_text": "Звоните",
            "booking_url": "",
            "dedup_source_text": "21 марта приглашаем на прогулку по Светлогорску, запись открыта.",
        },
        date_label="Сб, 21 марта, 11:30",
    )

    assert reason == "fallback"
    assert item is not None
    assert item["booking_text"] == "+7 921 710-11-61"
    assert item["booking_url"] == "tel:+79217101161"
    assert item["booking_line"] == "+7 921 710-11-61"


def test_apply_editorial_fallback_prefers_profile_guide_line_and_hides_raw_price_copy():
    item, reason = editorial.apply_editorial_fallback(
        {
            "canonical_title": "Экопрогулка в Южном парке",
            "digest_blurb": "Прогулка по Южному парку.",
            "summary_one_liner": "Прогулка по Южному парку.",
            "source_title": "Путешествия по пРуссии",
            "guide_names": ["Юлия Гришанова"],
            "guide_profile_facts": {
                "guide_line": "Юлия Гришанова, орнитолог и аккредитованный гид по Калининградской области"
            },
            "price_text": "500/300 руб взрослые/дети,пенсионеры",
            "booking_text": "@Yulia_Grishanova",
            "booking_url": "https://t.me/Yulia_Grishanova",
            "dedup_source_text": "22 марта приглашаем на экопрогулку в Южный парк, запись открыта.",
        },
        date_label="Вс, 22 марта, 09:00",
    )

    assert reason == "fallback"
    assert item is not None
    assert item["guide_line"] == "Юлия Гришанова, орнитолог и аккредитованный гид по Калининградской области"
    assert "price_line" not in item


@pytest.mark.asyncio
async def test_apply_digest_batch_copy_updates_only_title_and_blurb(monkeypatch):
    monkeypatch.setattr(digest_writer, "GUIDE_DIGEST_WRITER_ENABLED", True)

    captured_payload: list[dict] = []

    async def _fake_llm(payload_rows):
        captured_payload.extend(payload_rows)
        return {
            42: {
                "occurrence_id": 42,
                "title": "Экопрогулка в Южном парке",
                "digest_blurb": (
                    "Прогулка по Южному парку посвящена птицам и первым признакам весны. "
                    "Маршрут подойдёт тем, кто любит спокойные наблюдения в живой среде."
                ),
            }
        }

    monkeypatch.setattr(digest_writer, "_ask_digest_batch_llm", _fake_llm)

    rows = await digest_writer.apply_digest_batch_copy(
        [
            {
                "id": 42,
                "canonical_title": "Экопрогулка в Южном парке",
                "date": "2026-03-22",
                "time": "09:00",
                "status": "scheduled",
                "meeting_point": "Главный вход в Южный парк",
                "price_text": "500/300 руб взрослые/дети,пенсионеры",
                "booking_text": "@Yulia_Grishanova",
                "booking_url": "https://t.me/Yulia_Grishanova",
                "summary_one_liner": "Прогулка по Южному парку.",
                "digest_blurb": "Прогулка по Южному парку.",
                "guide_names": ["Юлия Гришанова"],
                "organizer_names": ["Путешествия по пРуссии"],
                "source_title": "Путешествия по пРуссии",
                "source_kind": "guide_personal",
                "source_about_text": "Запись: @Yulia_Grishanova",
                "source_about_links": ["https://t.me/Yulia_Grishanova"],
                "schedule_line": "Вс, 22 марта, 09:00",
                "guide_line": "Путешествия по пРуссии",
                "audience_line": "семьям, любителям природы",
                "meeting_point_line": "Главный вход в Южный парк",
                "price_line": "500/300 руб взрослые/дети, пенсионеры",
                "booking_line": "@Yulia_Grishanova",
                "audience_fit": ["семьям", "любителям природы"],
                "fact_pack": {
                    "canonical_title": "Экопрогулка в Южном парке",
                    "date": "2026-03-22",
                    "time": "09:00",
                    "meeting_point": "Главный вход в Южный парк",
                    "price_text": "500/300 руб взрослые/дети,пенсионеры",
                    "booking_url": "https://t.me/Yulia_Grishanova",
                    "guide_names": ["Юлия Гришанова"],
                    "audience_fit": ["семьям", "любителям природы"],
                    "summary_one_liner": "Прогулка по Южному парку с акцентом на птиц и вестников весны.",
                },
                "dedup_source_text": "22 марта приглашаем на прогулку по Южному парку, запись через @Yulia_Grishanova.",
            }
        ],
        family="new_occurrences",
        date_formatter=lambda date_iso, time_text: "Вс, 22 марта, 09:00",
    )

    assert len(captured_payload) == 1
    assert captured_payload[0]["fact_density"] in {"medium", "high"}
    assert captured_payload[0]["target_blurb_sentences"] in {2, 3}
    assert "booking" in captured_payload[0]["shell_fields_present"]
    assert "meeting_point" in captured_payload[0]["shell_fields_present"]

    row = rows[0]
    assert row["canonical_title"] == "Экопрогулка в Южном парке"
    assert row["guide_line"] == "Путешествия по пРуссии"
    assert row["schedule_line"] == "Вс, 22 марта, 09:00"
    assert row["audience_line"].startswith("семьям")
    assert row["meeting_point_line"] == "Главный вход в Южный парк"
    assert row["price_line"].startswith("500/300 руб")
    assert row["booking_line"] == "@Yulia_Grishanova"
    assert row["booking_text"] == "@Yulia_Grishanova"
    assert row["booking_url"] == "https://t.me/Yulia_Grishanova"
    assert row["digest_blurb"].count(".") >= 2
