from __future__ import annotations


def test_normalize_plaintext_paragraphs_unjams_inline_bullets() -> None:
    from smart_event_update import _normalize_plaintext_paragraphs

    raw = "-Длительность около 2 часов •Возрастное ограничение: 12+ •Сбор группы у входа"
    normalized = _normalize_plaintext_paragraphs(raw) or ""
    assert "•" not in normalized
    assert normalized.count("\n- ") >= 1

