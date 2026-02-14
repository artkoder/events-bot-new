from smart_event_update import EventCandidate, _candidate_has_event_anchors, _looks_like_promo_or_congrats


def test_promo_is_skipped() -> None:
    assert _looks_like_promo_or_congrats("Акция! Скидка 50% на билеты") is True
    assert _looks_like_promo_or_congrats("Промокод: THEATRE10") is True


def test_congrats_schedule_is_skipped() -> None:
    text = (
        "Поздравляем с днем рождения!\\n\\n"
        "Ближайшие спектакли:\\n"
        "05.02 | Лорд Фаунтлерой\\n"
        "13.02 | Дикарь\\n"
    )
    assert _looks_like_promo_or_congrats(text) is True


def test_regular_event_not_skipped() -> None:
    # A normal event announcement should not be caught by the promo filter.
    text = "13.02 в 19:00 спектакль «Дикарь» в драмтеатре."
    assert _looks_like_promo_or_congrats(text) is False


def test_event_anchors_use_source_text_even_when_excerpt_has_no_datetime() -> None:
    # short_description is not allowed to contain date/time by prompt design,
    # so anchors must also consider the full source_text.
    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-78172842_7020",
        source_text="22 февраля состоится вечер стендап-комедии. Ресторану «МонАми» исполняется 1 год.",
        title="Вечер стендап-комедии",
        date="2026-02-22",
        time="19:00",
        location_name="МонАми, Гвардейск",
        raw_excerpt="Камерный вечер стендап-комедии с приглашёнными комиками.",
    )
    assert _candidate_has_event_anchors(candidate) is True
