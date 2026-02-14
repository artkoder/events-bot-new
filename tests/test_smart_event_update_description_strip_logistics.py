from __future__ import annotations

from smart_event_update import EventCandidate, _strip_infoblock_logistics_from_description


def test_strip_infoblock_logistics_removes_obvious_duplicates():
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-78172842_7020",
        source_text="",
        title="Вечер стендап-комедии",
        date="2026-02-22",
        time="19:00",
        location_name="Ресторан Mon Ame",
        location_address="г. Гвардейск, ул. Юбилейная, д. 1а",
        city="Гвардейск",
        ticket_price_min=800,
        ticket_price_max=800,
        is_free=False,
    )

    text = (
        "В Гвардейске 22 февраля состоится вечер стендап-комедии с участием лучших комиков региона. "
        "Организаторы приглашают всех желающих насладиться живым юмором.\n\n"
        "Сбор гостей начнётся в 19:00 в ресторане «Mon Ame», расположенном по адресу: г. Гвардейск, ул. Юбилейная, д. 1а. "
        "Стоимость билета составляет 800 рублей. Посетителям предлагают забронировать столы по телефону 89622554547.\n\n"
        "Мероприятие предназначено для лиц старше 18 лет. Телефон для бронирования столов: 89622554547."
    )

    out = _strip_infoblock_logistics_from_description(text, candidate=candidate)
    assert out
    assert "\\1" not in out

    # No duplicated logistics from the quick facts infoblock.
    assert "22 февраля" not in out
    assert "19:00" not in out
    assert "Юбилейн" not in out
    assert "800" not in out
    assert "8962" not in out

    # Still keeps narrative content.
    assert "лучших комиков" in out
    assert "старше 18" in out


def test_strip_infoblock_logistics_drops_dangling_price_clause():
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text="",
        title="Тест",
        date="2026-02-22",
        time="19:00",
        location_name="Бар «Бастион»",
        location_address=None,
        city="Калининград",
        ticket_price_min=800,
        ticket_price_max=800,
        is_free=False,
    )

    text = (
        "В баре «Бастион» состоится вечеринка.\n\n"
        "Вход на концерт будет стоить 800 рублей. Девушкам вход свободный, а для мужчин стоимость посещения составит 800 рублей."
    )

    out = _strip_infoblock_logistics_from_description(text, candidate=candidate)
    assert out
    # Must not leave broken tails after removing the price.
    assert "составит" not in out


def test_strip_infoblock_logistics_keeps_ticket_condition_sentences():
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-48383763_39150",
        source_text="",
        title="Секрет пяти хвойнок",
        date="2026-02-15",
        time="11:00",
        location_name="Калининградский зоопарк",
        location_address=None,
        city="Калининград",
        ticket_price_min=500,
        ticket_price_max=500,
        is_free=False,
    )

    text = (
        "Экскурсия проходит на улице, поэтому одевайтесь теплее.\n\n"
        "Входной билет в зоопарк нужен дополнительно.\n\n"
        "Билеты доступны по ссылке: https://t.me/+7qdz... (необязательно)."
    )

    out = _strip_infoblock_logistics_from_description(text, candidate=candidate)
    assert out
    assert "входной билет" in out.lower()
    assert "t.me" not in out.lower()
    assert "по ссылке" not in out.lower()


def test_strip_infoblock_logistics_preserves_numbered_lists():
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-48383763_39150",
        source_text="",
        title="Секрет пяти хвойнок",
        date="2026-02-15",
        time="11:00",
        location_name="Калининградский зоопарк",
        location_address=None,
        city="Калининград",
        ticket_price_min=500,
        ticket_price_max=500,
        is_free=False,
    )

    text = (
        "### Условия\n\n"
        "1. Длительность прогулки около 2 часов.\n"
        "2. Большая часть экскурсии проходит на улице — одевайтесь теплее.\n"
        "3. Максимальная группа — 25 человек.\n\n"
        "Входной билет в зоопарк нужен дополнительно."
    )

    out = _strip_infoblock_logistics_from_description(text, candidate=candidate)
    assert out
    assert "### Условия" in out
    assert "\n2." in out
    assert "\n3." in out
