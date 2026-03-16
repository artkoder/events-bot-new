from __future__ import annotations

import pytest

from guide_excursions.digest import build_digest_messages
from guide_excursions import dedup as guide_dedup
from guide_excursions import digest_writer as guide_digest_writer


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
            "city": "Калининград",
            "audience_fit": ["местным", "туристам", "любителям истории"],
            "digest_blurb": "Прогулка по старому району с городскими сюжетами и биографиями.",
            "route_summary": "Закхаймские ворота, старые кварталы и городские истории",
            "duration_text": "2,5-3 часа",
            "meeting_point": "у Закхаймских ворот",
            "price_text": "2000 ₽",
            "booking_text": "@tanja",
            "booking_url": "https://t.me/tanja",
            "channel_url": "https://t.me/tanja_from_koenigsberg/3895",
            "popularity_mark": "❤️",
            "audience_region_line": "Будет интересно и тем, кто давно живёт в регионе, и тем, кто только с ним знакомится",
            "fact_pack": {"audience_region_fit_label": "mixed"},
        }
    ]
    messages = build_digest_messages(rows, family="new_occurrences")
    assert len(messages) == 1
    text = messages[0]
    assert "Новые экскурсии гидов" in text
    assert '1. ❤️ <a href="https://t.me/tanja_from_koenigsberg/3895">У Тани на районе: Закхайм и окрестности</a>' in text
    assert "Татьяна из Кёнигсберга" in text
    assert '<a href="https://t.me/tanja_from_koenigsberg">Татьяна из Кёнигсберга</a>' not in text
    assert "👤 Гид: Татьяна Удовенко" in text
    assert "🏙 Локация: Калининград" in text
    assert "🏠 Будет интересно и тем, кто давно живёт в регионе, и тем, кто только с ним знакомится" in text
    assert "👥 Кому подойдёт: местным, туристам, любителям истории" in text
    assert "🗺 Что в маршруте: Закхаймские ворота, старые кварталы и городские истории" in text
    assert "⏱ Продолжительность: 2,5-3 часа" in text
    assert "📍 Место сбора: у Закхаймских ворот" in text
    assert '✍️ Запись: <a href="https://t.me/tanja">@tanja</a>' in text
    assert "📣 Канал:" not in text


def test_build_digest_messages_omits_unknown_placeholders():
    rows = [
        {
            "id": 7,
            "canonical_title": "Южный Амалиенау",
            "source_title": "Хранители руин",
            "date": "2026-03-15",
            "time": "10:00",
            "digest_blurb": "Прогулка по району вилл и историям его жителей.",
            "seats_text": "Не определено",
            "booking_text": "@ruin_keepers_admin",
            "booking_url": "https://t.me/ruin_keepers_admin",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "Не определено" not in text
    assert "🎟" not in text


def test_build_digest_messages_omits_one_date_placeholder():
    rows = [
        {
            "id": 8,
            "canonical_title": "Судьбы и шедевры",
            "source_title": "Профи-тур",
            "date": "2026-03-26",
            "time": "10:30",
            "digest_blurb": "Групповая экскурсия по галерее и городу.",
            "seats_text": "одна дата",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "🎟" not in text
    assert "одна дата" not in text


def test_build_digest_messages_omits_audience_region_without_llm_line():
    rows = [
        {
            "id": 9,
            "canonical_title": "Южный парк",
            "source_title": "Путешествия по пРуссии",
            "date": "2026-03-22",
            "time": "09:00",
            "digest_blurb": "Экопрогулка по Южному парку.",
            "fact_pack": {"audience_region_fit_label": "mixed"},
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "🏠 " not in text


def test_build_digest_messages_omits_raw_price_copy_without_normalized_line():
    rows = [
        {
            "id": 10,
            "canonical_title": "Экопрогулка в Южном парке",
            "source_title": "Путешествия по пРуссии",
            "date": "2026-03-22",
            "time": "09:00",
            "digest_blurb": "Экопрогулка по Южному парку.",
            "price_text": "500/300 руб взрослые/дети,пенсионеры",
            "guide_profile_facts": {
                "guide_line": "Юлия Гришанова, орнитолог и аккредитованный гид по Калининградской области"
            },
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "💸" not in text
    assert "👤 Гид: Юлия Гришанова" in text


def test_build_digest_messages_normalizes_public_location_aliases():
    rows = [
        {
            "id": 11,
            "canonical_title": "Весенняя Роминта",
            "source_title": "Путешествия по пРуссии",
            "date": "2026-04-16",
            "digest_blurb": "Маршрут по лесным дорогам и весенним точкам бывшей Роминты.",
            "city": "Роминта",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "🏙 Локация: Краснолесье и окрестности Роминтенской пущи" in text


def test_build_digest_messages_adds_visual_separator_between_cards():
    rows = [
        {
            "id": 12,
            "canonical_title": "Первая экскурсия",
            "source_title": "Канал 1",
            "date": "2026-03-22",
            "digest_blurb": "Первая карточка.",
        },
        {
            "id": 13,
            "canonical_title": "Вторая экскурсия",
            "source_title": "Канал 2",
            "date": "2026-03-23",
            "digest_blurb": "Вторая карточка.",
        },
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "──────────" in text


def test_build_digest_messages_prefers_llm_emoji_and_plural_guide_label():
    rows = [
        {
            "id": 14,
            "canonical_title": "Экопрогулка в Южном парке",
            "source_title": "Путешествия по пРуссии",
            "source_post_url": "https://t.me/amber_fringilla/5806",
            "date": "2026-03-22",
            "digest_blurb": "Прогулка по Южному парку.",
            "guide_names": ["Юлия Гришанова", "Анастасия Туз"],
            "lead_emoji": "🐦",
            "popularity_mark": "❤️",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert '1. 🐦 <a href="https://t.me/amber_fringilla/5806">Экопрогулка в Южном парке</a>' in text
    assert "👥 Гиды: Юлия Гришанова, Анастасия Туз" in text


def test_build_digest_messages_renders_organizer_when_guide_unknown():
    rows = [
        {
            "id": 15,
            "canonical_title": "Третьяковская галерея и история Калининграда",
            "source_title": "Профи-тур",
            "source_kind": "excursion_operator",
            "date": "2026-03-26",
            "digest_blurb": "Маршрут объединяет выставку и городскую историю.",
            "organizer_line": "Профи-тур, организатор авторских экскурсий по Калининграду",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "🏢 Организатор: Профи-тур, организатор авторских экскурсий по Калининграду" in text
    assert "👤 Гид:" not in text


def test_build_digest_messages_demotes_operator_like_guide_line_to_organizer():
    rows = [
        {
            "id": 16,
            "canonical_title": "Осетровая симфония",
            "source_title": "Экскурсии от «Профи-тур»",
            "source_kind": "excursion_operator",
            "date": "2026-03-22",
            "digest_blurb": "Дегустационная поездка с осетровыми специалитетами.",
            "guide_line": "Профи-тур: организация интересных экскурсий и путешествий по Калининграду",
        }
    ]
    text = build_digest_messages(rows, family="new_occurrences")[0]
    assert "👤 Гид:" not in text
    assert "🏢 Организатор:" in text


def test_digest_writer_retry_after_parser_reads_provider_hint():
    assert guide_digest_writer._retry_after_seconds("Rate limit exceeded: tpm (retry after 14000ms)") == 14.0
    assert guide_digest_writer._retry_after_seconds("plain failure") is None


@pytest.mark.asyncio
async def test_digest_writer_passes_candidate_key_ids_to_google_ai():
    class _FakeClient:
        async def generate_content_async(self, **kwargs):
            assert kwargs["candidate_key_ids"] == ["guide-key2-id"]
            return ('{"items":[{"occurrence_id":1,"title":"Тестовая прогулка","digest_blurb":"Короткое описание."}]}', None)

    result = await guide_digest_writer._ask_digest_batch_llm_batch(
        _FakeClient(),
        [{"occurrence_id": 1, "title_seed": "Тестовая прогулка", "fact_pack": {}}],
        candidate_key_ids=["guide-key2-id"],
    )

    assert result == {
        1: {
            "occurrence_id": 1,
            "title": "Тестовая прогулка",
            "digest_blurb": "Короткое описание.",
        }
    }


@pytest.mark.asyncio
async def test_apply_digest_batch_copy_materializes_public_lines_from_llm(monkeypatch):
    async def _fake_llm(payload_rows):
        assert payload_rows[0]["source_about_excerpt"]
        assert payload_rows[0]["guide_profile_facts"]["guide_line"] == "Юлия Гришанова, аккредитованный гид и орнитолог"
        return {
            1: {
                "occurrence_id": 1,
                "lead_emoji": "🐦",
                "title": "Экопрогулка в Южном парке",
                "digest_blurb": "Прогулка по Южному парку собирает наблюдения за птицами и первыми весенними изменениями.",
                "guide_line": "Юлия Гришанова, аккредитованный гид и орнитолог",
                "price_line": "500 ₽ для взрослых, 300 ₽ для детей и пенсионеров",
                "route_line": "птицы Южного парка и первые весенние цветы",
                "audience_region_line": "Экскурсия будет интересна и местным, и гостям региона",
            }
        }

    monkeypatch.setattr(guide_digest_writer, "_ask_digest_batch_llm", _fake_llm)
    rows = [
        {
            "id": 1,
            "canonical_title": "Экопрогулка в Южном парке",
            "date": "2026-03-22",
            "time": "09:00",
            "city": "Калининград",
            "price_text": "500/300 руб взрослые/дети,пенсионеры",
            "dedup_source_text": "Экопрогулка по Южному парку с птицами и первыми весенними цветами.",
            "source_about_text": "Юлия Гришанова, орнитолог, аккредитованный гид по Калининградской области.",
            "guide_profile_summary": "Аккредитованный гид и орнитолог.",
            "guide_profile_facts": {"guide_line": "Юлия Гришанова, аккредитованный гид и орнитолог"},
            "fact_pack": {
                "main_hook": "Прогулка по Южному парку с фокусом на весенние птицы и первые цветы",
                "audience_region_fit_label": "mixed",
            },
        }
    ]
    out = await guide_digest_writer.apply_digest_batch_copy(rows, family="new_occurrences", date_formatter=lambda d, t: "Вс, 22 марта, 09:00")
    assert out[0]["lead_emoji"] == "🐦"
    assert out[0]["guide_line"] == "Юлия Гришанова, аккредитованный гид и орнитолог"
    assert out[0]["price_line"] == "500 ₽ для взрослых, 300 ₽ для детей и пенсионеров"
    assert out[0]["route_line"] == "птицы Южного парка и первые весенние цветы"
    assert out[0]["audience_region_line"] == "Экскурсия будет интересна и местным, и гостям региона"


@pytest.mark.asyncio
async def test_apply_digest_batch_copy_rejects_rawish_price_and_generic_audience(monkeypatch):
    async def _fake_llm(payload_rows):
        return {
            1: {
                "occurrence_id": 1,
                "title": "Экопрогулка в Южном парке",
                "digest_blurb": "Прогулка по Южному парку с наблюдением за птицами и первыми весенними изменениями.",
                "guide_line": "Юлия Гришанова",
                "price_line": "500/300 руб взрослые/дети,пенсионеры",
                "route_line": "птицы Южного парка и первые весенние цветы",
                "audience_region_line": "и для жителей региона, и для гостей",
            }
        }

    monkeypatch.setattr(guide_digest_writer, "_ask_digest_batch_llm", _fake_llm)
    rows = [
        {
            "id": 1,
            "canonical_title": "Экопрогулка в Южном парке",
            "date": "2026-03-22",
            "time": "09:00",
            "city": "Калининград",
            "price_text": "500/300 руб взрослые/дети,пенсионеры",
            "dedup_source_text": "Экопрогулка по Южному парку с птицами и первыми весенними цветами.",
            "fact_pack": {
                "main_hook": "Прогулка по Южному парку с фокусом на весенние птицы и первые цветы",
                "audience_region_fit_label": "mixed",
            },
        }
    ]
    out = await guide_digest_writer.apply_digest_batch_copy(
        rows,
        family="new_occurrences",
        date_formatter=lambda d, t: "Вс, 22 марта, 09:00",
    )
    assert "price_line" not in out[0]
    assert "audience_region_line" not in out[0]


@pytest.mark.asyncio
async def test_apply_digest_batch_copy_rejects_blurb_that_duplicates_region_fit(monkeypatch):
    async def _fake_llm(payload_rows):
        return {
            1: {
                "occurrence_id": 1,
                "title": "Историко-архитектурная прогулка «Как это построено»",
                "digest_blurb": "Прогулка по Хуфену будет интересна и местным, и гостям региона.",
                "audience_region_line": "Подойдёт и тем, кто уже живёт в регионе, и тем, кто только его открывает",
            }
        }

    monkeypatch.setattr(guide_digest_writer, "_ask_digest_batch_llm", _fake_llm)
    rows = [
        {
            "id": 1,
            "canonical_title": "Историко-архитектурная прогулка «Как это построено»",
            "summary_one_liner": "Новая историко-архитектурная прогулка по Хуфену.",
            "digest_blurb": "Новая историко-архитектурная прогулка по Хуфену.",
            "fact_pack": {"audience_region_fit_label": "mixed"},
        }
    ]
    out = await guide_digest_writer.apply_digest_batch_copy(rows, family="new_occurrences", date_formatter=lambda d, t: None)
    assert out[0]["digest_blurb"] == "Новая историко-архитектурная прогулка по Хуфену."
    assert out[0]["audience_region_line"] == "Подойдёт и тем, кто уже живёт в регионе, и тем, кто только его открывает"


def test_accept_blurb_candidate_rejects_term_family_conflict():
    row = {
        "canonical_title": "Историко-архитектурная прогулка «Как это построено»",
        "fact_pack": {},
    }
    assert not guide_digest_writer._accept_blurb_candidate(
        "Экскурсия раскрывает историю и архитектуру района Хуфен.",
        row,
        date_label=None,
    )


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
