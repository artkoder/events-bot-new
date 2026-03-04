from __future__ import annotations

from smart_event_update import _strip_channel_promo_from_description, _strip_promo_lines


def test_deterministic_promo_strippers_are_noop_now() -> None:
    # Policy: text shaping is handled by LLM. Deterministic regex cutters must not
    # delete meaning-bearing text.
    src = (
        "Секрет пяти хвойнок — встреча в зоопарке.\n"
        "Информация о событиях в зоопарке доступна в Telegram-канале «АНОНС 39»: https://t.me/+7qdzwqLJPiEyY2Fi.\n"
        "Начало в 18:00.\n"
    )
    out = _strip_promo_lines(src) or ""
    assert out == src.strip()

    text = (
        "Это встреча для всех, кому интересно узнать больше.\n\n"
        "Информация о событиях в зоопарке доступна в Telegram-канале «АНОНС 39»: https://t.me/+7qdzwqLJPiEyY2Fi.\n\n"
        "В программе — рассказ и обсуждение."
    )
    out = _strip_channel_promo_from_description(text) or ""
    assert out == text.strip()
