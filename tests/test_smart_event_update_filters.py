from smart_event_update import _looks_like_promo_or_congrats


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

