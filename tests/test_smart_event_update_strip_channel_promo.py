from __future__ import annotations

from smart_event_update import _strip_promo_lines, _strip_channel_promo_from_description


def test_strip_promo_lines_drops_generic_announcement_channel_line():
    src = (
        "Секрет пяти хвойнок — встреча в зоопарке.\n"
        "Информация о событиях в зоопарке доступна в Telegram-канале «АНОНС 39»: https://t.me/+7qdzwqLJPiEyY2Fi.\n"
        "Начало в 18:00.\n"
    )
    out = _strip_promo_lines(src) or ""
    assert "АНОНС 39" not in out
    assert "t.me/+7qdzwqLJPiEyY2Fi" not in out
    assert "Начало" in out


def test_strip_channel_promo_from_description_removes_sentence():
    text = (
        "Это встреча для всех, кому интересно узнать больше.\n\n"
        "Информация о событиях в зоопарке доступна в Telegram-канале «АНОНС 39»: https://t.me/+7qdzwqLJPiEyY2Fi.\n\n"
        "В программе — рассказ и обсуждение."
    )
    out = _strip_channel_promo_from_description(text) or ""
    assert "АНОНС 39" not in out
    assert "t.me/+7qdzwqLJPiEyY2Fi" not in out
    assert "В программе" in out


def test_strip_channel_promo_from_description_keeps_start_time_prefix_when_appended():
    text = (
        "Начало в 18:00, следите за анонсами в Telegram-канале «АНОНС 39»: https://t.me/+7qdzwqLJPiEyY2Fi."
    )
    out = _strip_channel_promo_from_description(text) or ""
    assert "t.me/+7qdzwqLJPiEyY2Fi" not in out
    assert "АНОНС 39" not in out
    assert "Начало" in out
    assert "18:00" in out
