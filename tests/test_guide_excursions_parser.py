from __future__ import annotations

from datetime import datetime, timezone

from guide_excursions.editorial import apply_editorial_fallback, build_booking_candidates
from guide_excursions.parser import parse_post_occurrences


def test_parse_single_aggregator_excursion():
    text = (
        '🌿 14 марта в 11:00 "У Тани на районе: Закхайм и окрестности" '
        "Авторская экскурсия Татьяны Удовенко\n"
        "☎️Запись по телефону с 08.30-20.30 521-161 или + 7 921-710-11-61\n"
        "В самом центре современного Калининграда есть интересный исторический район."
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 13, 8, 0, tzinfo=timezone.utc),
        source_kind="aggregator",
        source_title="В Калининграде",
        channel_url="https://t.me/vkaliningrade/4584",
        fallback_guide_name="В Калининграде",
    )
    assert len(occurrences) == 1
    item = occurrences[0]
    assert item.canonical_title == "У Тани на районе: Закхайм и окрестности"
    assert item.date_iso == "2026-03-14"
    assert item.time_text == "11:00"
    assert item.digest_eligible is True
    assert item.guide_names[0] in {"Татьяны Удовенко", "В Калининграде"}
    assert item.booking_text


def test_parse_multi_schedule_lines():
    text = (
        "Расписание экскурсий на март\n"
        '15.03 Вс "На Восток-2" уже набрана\n'
        '22.03 Вс "МК по Ратсхофу"\n'
        '29.03 Вс МК "Марауненхоф, ч.I".'
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 2, 24, 9, 0, tzinfo=timezone.utc),
        source_kind="guide_personal",
        source_title="alev701",
        channel_url="https://t.me/alev701/631",
        fallback_guide_name="Алексей А.",
    )
    assert len(occurrences) == 3
    assert [item.date_iso for item in occurrences] == ["2026-03-15", "2026-03-22", "2026-03-29"]
    assert occurrences[0].canonical_title == "На Восток-2"
    assert occurrences[1].canonical_title == "МК по Ратсхофу"
    assert occurrences[2].canonical_title == "Марауненхоф, ч.I"
    assert occurrences[0].digest_eligible is False
    assert occurrences[0].digest_eligibility_reason == "closed_without_booking"


def test_parse_amber_fringilla_multi_excursion_post_materializes_multiple_occurrences():
    text = (
        "Экскурсии и путешествия на март-апрель.\n"
        "22 марта, воскресенье, в 9:00 экопрогулка в Южный парк. Узнаем о птицах и вестниках весны.\n"
        "Продолжительность: 1,5-2 часа.\n"
        "Встречаемся у главного входа в Южный Парк.\n"
        "Стоимость 500/300 руб взрослые/дети, пенсионеры.\n"
        "Запись @Yulia_Grishanova.\n"
        "\n"
        "5 апреля, воскресенье, в 10:00 знакомство с историей растительного мира на острове Канта.\n"
        "Продолжительность около 2 часов.\n"
        "Стоимость 800 руб.\n"
        "Запись @Yulia_Grishanova.\n"
        "\n"
        "10 апреля, пятница. Самбия: от 0 до 60 м.\n"
        "Подробности позже.\n"
        "Запись @Yulia_Grishanova.\n"
        "\n"
        "16 апреля, четверг. Весенняя буковая роща.\n"
        "Пройдусь с вами от Ново-Московского до Ладушкина.\n"
        "Стоимость 1000 руб + билеты.\n"
        "Запись @Yulia_Grishanova.\n"
        "\n"
        "26 апреля, воскресенье. Весенняя Роминта.\n"
        "Подробности будут позже.\n"
        "Запись @Yulia_Grishanova.\n"
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc),
        source_kind="guide_personal",
        source_title="Путешествия по пРуссии",
        channel_url="https://t.me/amber_fringilla/5806",
        fallback_guide_name="Юлия Гришанова",
    )
    assert len(occurrences) == 5
    assert [item.date_iso for item in occurrences] == [
        "2026-03-22",
        "2026-04-05",
        "2026-04-10",
        "2026-04-16",
        "2026-04-26",
    ]
    assert occurrences[0].canonical_title == "Экопрогулка в Южный парк"
    assert occurrences[0].duration_text is None
    assert occurrences[0].meeting_point == "у главного входа в Южный Парк"
    assert occurrences[0].booking_text == "@Yulia_Grishanova"
    assert occurrences[3].canonical_title == "Весенняя буковая роща"
    assert occurrences[3].route_summary is None
    assert occurrences[3].group_format is None
    assert occurrences[4].canonical_title == "Весенняя Роминта"
    assert occurrences[4].time_text is None


def test_reflection_post_is_not_digest_eligible():
    text = (
        "У меня на трипстере стоит единственная экскурсия «Куршская коса с профессиональным орнитологом».\n"
        "Она у меня уже 2 год месяца на 3-4 вперед закрыта.\n"
        "Заказчик пропал."
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc),
        source_kind="guide_personal",
        source_title="Amber Fringilla",
        channel_url="https://t.me/amber_fringilla/5762",
        fallback_guide_name="Amber Fringilla",
    )
    assert len(occurrences) == 1
    assert occurrences[0].digest_eligible is False
    assert occurrences[0].digest_eligibility_reason == "context_only"


def test_operational_departure_post_is_not_digest_eligible():
    text = (
        'Завтра  в 09:00 от ростверка ДС Центральной площади, отправление экспресса автобуса Higer '
        'т105вс39, серый, в путешествие на "Восток-2". Предположительное возвращение в 20:00.'
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 14, 18, 0, tzinfo=timezone.utc),
        source_kind="guide_personal",
        source_title="alev701",
        channel_url="https://t.me/alev701/636",
        fallback_guide_name="Алексей А.",
    )
    assert len(occurrences) == 1
    assert occurrences[0].canonical_title == "Восток-2"
    assert occurrences[0].digest_eligible is False
    assert occurrences[0].digest_eligibility_reason == "operational_only"


def test_profitour_title_keeps_travel_prefix():
    text = (
        "15.03.2026г.\n"
        "В 11.00 выезд от Дома Советов.\n"
        "Приглашаем в путешествие на ферму по разведению осетра и улиток.\n"
        "Стоимость: 2500руб.\n"
        "Количество мест ограничено!"
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc),
        source_kind="excursion_operator",
        source_title="Профи-тур",
        channel_url="https://t.me/excursions_profitour/857",
        fallback_guide_name="Профи-тур",
    )
    assert len(occurrences) == 1
    assert occurrences[0].canonical_title == "Путешествие на ферму по разведению осетра и улиток"


def test_gid_zelenogradsk_march_block_does_not_take_april_premiere_title():
    text = (
        "🔹 18 марта в 12:00\n"
        "Расширенная экскурсия по Зеленоградску (для экскурсоводов и увлечённых слушателей)\n"
        "⏳ Продолжительность: 4+ часа\n"
        "💰 Стоимость: 1000₽\n"
        "· 18 марта: мест нет, только в лист ожидания.\n"
        "🎉 Апрельская премьера:\n"
        "Вас ждёт премьера моей новой экскурсии «Гранц - Нахимовск - Зеленоградск: как всё начиналось».\n"
        "📲 http://t.me/gid_zelenogradsk_kotova_natalia\n"
    )
    occurrences = parse_post_occurrences(
        text=text,
        post_date=datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc),
        source_kind="guide_personal",
        source_title="Котова Наталья",
        channel_url="https://t.me/gid_zelenogradsk/2684",
        fallback_guide_name="Котова Наталья",
    )
    assert len(occurrences) >= 1
    digest_items = [item for item in occurrences if item.digest_eligible]
    assert len(digest_items) == 1
    assert digest_items[0].canonical_title == "Расширенная экскурсия по Зеленоградску (для экскурсоводов и увлечённых слушателей)"
    assert digest_items[0].booking_url == "http://t.me/gid_zelenogradsk_kotova_natalia"


def test_source_about_booking_candidates_prefer_telegram_contact():
    row = {
        "source_about_text": (
            "По вопросам участия: +7 962 255-54-91 - Татьяна "
            "http://t.me/tatamartynyuk"
        ),
        "source_about_links": ["http://t.me/tatamartynyuk", "https://profi-tur39.ru/"],
    }
    candidates = build_booking_candidates(row)
    assert candidates[0]["text"] == "@tatamartynyuk"
    assert candidates[0]["url"] == "https://t.me/tatamartynyuk"
    assert any(item["url"].startswith("tel:") for item in candidates)


def test_editorial_fallback_neutralizes_relative_blurb():
    row = {
        "id": 30,
        "canonical_title": "Восток-2",
        "digest_blurb": 'Завтра в 09:00 от ростверка ДС Центральной площади отправление в путешествие на "Восток-2".',
        "summary_one_liner": 'Завтра в 09:00 от ростверка ДС Центральной площади отправление в путешествие на "Восток-2".',
        "time": "09:00",
        "dedup_source_text": 'Завтра в 09:00 от ростверка ДС Центральной площади, отправление экспресса автобуса Higer.',
        "booking_text": "@guide",
        "booking_url": "https://t.me/guide",
        "source_about_text": "",
        "source_about_links": [],
    }
    refined, reason = apply_editorial_fallback(row, date_label="Вс, 15 марта, 09:00")
    assert reason == "fallback"
    assert refined is not None
    assert "Завтра" not in refined["digest_blurb"]
    assert "Вс, 15 марта, 09:00" in refined["digest_blurb"]
