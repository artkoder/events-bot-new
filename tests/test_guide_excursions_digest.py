from __future__ import annotations

import pytest

from guide_excursions.digest import build_digest_messages
from guide_excursions import dedup as guide_dedup


def test_build_digest_messages_renders_cards():
    rows = [
        {
            "id": 1,
            "canonical_title": "У Тани на районе: Закхайм и окрестности",
            "guide_names": ["Татьяна Удовенко"],
            "source_title": "Татьяна из Кёнигсберга",
            "source_username": "tanja_from_koenigsberg",
            "date": "2026-03-14",
            "time": "11:00",
            "audience_fit": ["местным", "туристам", "любителям истории"],
            "digest_blurb": "Прогулка по старому району с городскими сюжетами и биографиями.",
            "meeting_point": "у Закхаймских ворот",
            "price_text": "2000 ₽",
            "booking_text": "@tanja",
            "booking_url": "https://t.me/tanja",
            "channel_url": "https://t.me/tanja_from_koenigsberg/3895",
            "popularity_mark": "❤️",
        }
    ]
    messages = build_digest_messages(rows, family="new_occurrences")
    assert len(messages) == 1
    text = messages[0]
    assert "Новые экскурсии гидов" in text
    assert '1. ❤️ <a href="https://t.me/tanja_from_koenigsberg/3895">У Тани на районе: Закхайм и окрестности</a>' in text
    assert '<a href="https://t.me/tanja_from_koenigsberg">Татьяна из Кёнигсберга</a>' in text
    assert "👥 Для кого: местным, туристам, любителям истории" in text
    assert '✍️ Запись: <a href="https://t.me/tanja">@tanja</a>' in text
    assert "📣 Канал:" not in text


@pytest.mark.asyncio
async def test_deduplicate_occurrence_rows_merges_focus_teaser(monkeypatch):
    monkeypatch.setattr(guide_dedup, "GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED", False)
    rows = [
        {
            "id": 24,
            "_score": 1.4,
            "canonical_title": "Два легендарных музея в одном путешествии!",
            "summary_one_liner": "Мамоновский городской музей и музей кирпича в рамках нашего путешествия на юго-запад области.",
            "date": "2026-03-14",
            "time": "",
            "booking_url": "https://t.me/ruin_keepers_admin",
            "booking_text": "@ruin_keepers_admin",
            "price_text": "",
            "meeting_point": "",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": ["детям", "любителям истории"],
        },
        {
            "id": 25,
            "_score": 1.2,
            "canonical_title": "юго-запад области в Мамоново и к легендарному замку Бальга",
            "summary_one_liner": "14 марта приглашаем в путешествие на юго-запад области в Мамоново и к легендарному замку Бальга.",
            "date": "2026-03-14",
            "time": "",
            "booking_url": "",
            "booking_text": "",
            "price_text": "",
            "meeting_point": "",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": ["детям", "любителям истории", "любителям природы"],
        },
        {
            "id": 30,
            "_score": 1.1,
            "canonical_title": "Совсем другая экскурсия",
            "summary_one_liner": "Отдельный маршрут на другую дату.",
            "date": "2026-03-15",
            "time": "",
            "booking_url": "",
            "booking_text": "",
            "price_text": "",
            "meeting_point": "",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": [],
        },
    ]
    result = await guide_dedup.deduplicate_occurrence_rows(rows, family="new_occurrences", limit=24)
    assert [row["id"] for row in result.display_rows] == [25, 30]
    assert result.covered_occurrence_ids == [24, 25, 30]
    assert result.suppressed_occurrence_ids == [24]


@pytest.mark.asyncio
async def test_deduplicate_occurrence_rows_keeps_same_contact_distinct(monkeypatch):
    monkeypatch.setattr(guide_dedup, "GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED", False)
    rows = [
        {
            "id": 99,
            "_score": 1.1,
            "canonical_title": "Хаусмарки, барельефы, арки",
            "summary_one_liner": "Пешеходная экскурсия по Центральному району.",
            "date": "2026-03-28",
            "time": "11:00",
            "booking_url": "tel:08302030521",
            "booking_text": "521-...",
            "price_text": "",
            "meeting_point": "",
            "source_username": "vkaliningrade",
            "source_title": "Народный экскурсовод - Калининград",
            "source_kind": "aggregator",
            "audience_fit": [],
        },
        {
            "id": 100,
            "_score": 1.0,
            "canonical_title": "Анастасии Туз ☎️Запись по телефону с 08",
            "summary_one_liner": "Экскурсия про стрит-арт и субкультуру граффити.",
            "date": "2026-03-28",
            "time": "11:00",
            "booking_url": "tel:08302030521",
            "booking_text": "521-...",
            "price_text": "",
            "meeting_point": "",
            "source_username": "vkaliningrade",
            "source_title": "Народный экскурсовод - Калининград",
            "source_kind": "aggregator",
            "audience_fit": [],
        },
    ]
    result = await guide_dedup.deduplicate_occurrence_rows(rows, family="new_occurrences", limit=24)
    assert [row["id"] for row in result.display_rows] == [99, 100]
    assert result.suppressed_occurrence_ids == []


@pytest.mark.asyncio
async def test_deduplicate_occurrence_rows_merges_same_post_generic_title(monkeypatch):
    monkeypatch.setattr(guide_dedup, "GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED", False)
    rows = [
        {
            "id": 22,
            "_score": 1.2,
            "canonical_title": "Южный Амалиенау. История района в судьбах людей",
            "summary_one_liner": "15 марта приглашаем на прогулку по знаменитому району вилл.",
            "dedup_source_text": "15 марта приглашаем на прогулку «Хранителей руин» по знаменитому району вилл. Южный Амалиенау. История района в судьбах людей.",
            "date": "2026-03-15",
            "time": "10:00",
            "booking_url": "https://t.me/ruin_keepers_admin",
            "booking_text": "@ruin_keepers_admin",
            "price_text": "1000 рублей",
            "meeting_point": "",
            "channel_url": "https://t.me/ruin_keepers/5065",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": [],
        },
        {
            "id": 23,
            "_score": 1.0,
            "canonical_title": "Хранители руин",
            "summary_one_liner": "Прогулки Хранителей — это краеведческие экскурсии по городу и области.",
            "dedup_source_text": "15 марта приглашаем на прогулку «Хранителей руин» по знаменитому району вилл. Южный Амалиенау. История района в судьбах людей. Прогулки Хранителей — это краеведческие экскурсии.",
            "date": "2026-03-15",
            "time": "10:00",
            "booking_url": "https://t.me/ruin_keepers_admin",
            "booking_text": "@ruin_keepers_admin",
            "price_text": "1000 рублей",
            "meeting_point": "",
            "channel_url": "https://t.me/ruin_keepers/5065",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": [],
        },
    ]
    result = await guide_dedup.deduplicate_occurrence_rows(rows, family="new_occurrences", limit=24)
    assert [row["id"] for row in result.display_rows] == [22]
    assert result.covered_occurrence_ids == [22, 23]
    assert result.suppressed_occurrence_ids == [23]


@pytest.mark.asyncio
async def test_deduplicate_occurrence_rows_merges_same_day_teaser_update(monkeypatch):
    monkeypatch.setattr(guide_dedup, "GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED", False)
    rows = [
        {
            "id": 19,
            "_score": 0.9,
            "canonical_title": "город-сад",
            "summary_one_liner": "Больше интересных сведений можно узнать уже завтра, на прогулке Хранителей.",
            "dedup_source_text": "В преддверии прогулки по Амалиенау, которая состоится завтра, у вас еще есть возможность на нее записаться. Больше интересных сведений о районе город-сад можно узнать уже завтра.",
            "date": "2026-03-15",
            "time": "",
            "booking_url": "",
            "booking_text": "",
            "price_text": "",
            "meeting_point": "",
            "channel_url": "https://t.me/ruin_keepers/5075",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": [],
        },
        {
            "id": 22,
            "_score": 1.2,
            "canonical_title": "Южный Амалиенау. История района в судьбах людей",
            "summary_one_liner": "15 марта приглашаем на прогулку по знаменитому району вилл.",
            "dedup_source_text": "15 марта, в воскресенье, приглашаем на прогулку Хранителей по знаменитому району вилл. Южный Амалиенау.",
            "date": "2026-03-15",
            "time": "10:00",
            "booking_url": "https://t.me/ruin_keepers_admin",
            "booking_text": "@ruin_keepers_admin",
            "price_text": "1000 рублей",
            "meeting_point": "",
            "channel_url": "https://t.me/ruin_keepers/5065",
            "source_username": "ruin_keepers",
            "source_title": "Хранители руин | Ruin Keepers",
            "source_kind": "organization_with_tours",
            "audience_fit": [],
        },
    ]
    result = await guide_dedup.deduplicate_occurrence_rows(rows, family="new_occurrences", limit=24)
    assert [row["id"] for row in result.display_rows] == [22]
    assert result.covered_occurrence_ids == [19, 22]
    assert result.suppressed_occurrence_ids == [19]


@pytest.mark.asyncio
async def test_deduplicate_occurrence_rows_merges_schedule_rollup_with_departure_update(monkeypatch):
    monkeypatch.setattr(guide_dedup, "GUIDE_EXCURSIONS_DEDUP_LLM_ENABLED", False)
    rows = [
        {
            "id": 32,
            "_score": 1.1,
            "canonical_title": "На Восток-2",
            "summary_one_liner": "Расписание экскурсий на март 15.03 Вс \"На Восток-2\" уже набрана",
            "dedup_source_text": "Расписание экскурсий на март. 15.03 Вс На Восток-2 уже набрана.",
            "date": "2026-03-15",
            "time": "",
            "booking_url": "",
            "booking_text": "",
            "price_text": "",
            "meeting_point": "",
            "channel_url": "https://t.me/alev701/631",
            "source_username": "alev701",
            "source_title": "Андрей Левченков. Истории и прогулки",
            "source_kind": "guide_project",
            "audience_fit": [],
        },
        {
            "id": 30,
            "_score": 1.3,
            "canonical_title": "Восток-2",
            "summary_one_liner": "Завтра в 09:00 от ростверка ДС Центральной площади отправление в путешествие на Восток-2.",
            "dedup_source_text": "Завтра в 09:00 от ростверка ДС Центральной площади, отправление в путешествие на Восток-2. Предположительное возвращение в 20:00.",
            "date": "2026-03-15",
            "time": "09:00",
            "booking_url": "",
            "booking_text": "",
            "price_text": "",
            "meeting_point": "ростверк ДС Центральной площади",
            "channel_url": "https://t.me/alev701/636",
            "source_username": "alev701",
            "source_title": "Андрей Левченков. Истории и прогулки",
            "source_kind": "guide_project",
            "audience_fit": [],
        },
    ]
    result = await guide_dedup.deduplicate_occurrence_rows(rows, family="new_occurrences", limit=24)
    assert [row["id"] for row in result.display_rows] == [30]
    assert result.covered_occurrence_ids == [32, 30]
    assert result.suppressed_occurrence_ids == [32]
