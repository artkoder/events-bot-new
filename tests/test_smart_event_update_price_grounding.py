import smart_event_update as su


def test_has_price_evidence_accepts_ruble_and_number() -> None:
    text = "Билеты: 1500 ₽, купить можно по ссылке."
    assert su._has_price_evidence(text, 1500) is True


def test_has_price_evidence_rejects_when_no_price_context() -> None:
    text = "24 апреля в 19:00 состоится концерт в Светлогорске."
    assert su._has_price_evidence(text, 1500) is False


def test_has_price_evidence_rejects_when_context_without_numbers() -> None:
    text = "Стоимость уточняйте в кассе."
    assert su._has_price_evidence(text, 1500) is False


def test_has_price_evidence_rejects_when_numbers_without_context() -> None:
    text = "Начало в 19:00, 24 апреля."
    assert su._has_price_evidence(text, 1500) is False

