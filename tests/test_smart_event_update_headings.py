from __future__ import annotations


def test_ensure_minimal_description_headings_inserts_when_missing() -> None:
    from smart_event_update import _ensure_minimal_description_headings

    text = (
        "Короткий лид без заголовка.\n\n"
        + " ".join(["Детали"] * 120)
    )
    out = _ensure_minimal_description_headings(text) or ""
    assert out != text.strip()
    assert "### О событии" in out
    assert out.startswith("Короткий лид без заголовка.")


def test_ensure_minimal_description_headings_inserts_for_short_two_paras() -> None:
    from smart_event_update import _ensure_minimal_description_headings

    text = "Лид.\n\nДетали."
    out = _ensure_minimal_description_headings(text) or ""
    assert out != text.strip()
    assert out.splitlines()[0].strip() == "Лид."
    assert "### О событии" in out


def test_ensure_minimal_description_headings_keeps_single_paragraph() -> None:
    from smart_event_update import _ensure_minimal_description_headings

    text = "Один абзац без заголовка."
    out = _ensure_minimal_description_headings(text) or ""
    assert out == text.strip()


def test_ensure_minimal_description_headings_does_not_insert_when_heading_present() -> None:
    from smart_event_update import _ensure_minimal_description_headings

    text = (
        "Короткий лид без заголовка.\n\n"
        "### Программа\n"
        "- Пункт 1\n"
        "- Пункт 2\n\n"
        "Финальная ремарка."
    )
    out = _ensure_minimal_description_headings(text) or ""
    assert out == text.strip()
    assert "### Подробности" not in out


def test_normalize_plaintext_paragraphs_keeps_heading_with_inline_body() -> None:
    from smart_event_update import _normalize_plaintext_paragraphs

    text = (
        "### Подробности\n"
        "Короткий абзац под заголовком.\n\n"
        "### Программа\n"
        "- Пункт 1\n"
        "- Пункт 2"
    )
    out = _normalize_plaintext_paragraphs(text) or ""
    assert "### Подробности" in out
    assert "Короткий абзац под заголовком." in out
    assert "### Программа" in out
    assert "- Пункт 1" in out
    assert "- Пункт 2" in out


def test_normalize_plaintext_paragraphs_drops_orphan_heading_before_same_level_heading() -> None:
    from smart_event_update import _normalize_plaintext_paragraphs

    text = "### Подробности\n\n### Формат турнира\nТекст про формат."
    out = _normalize_plaintext_paragraphs(text) or ""
    assert "### Подробности" not in out
    assert out.startswith("### Формат турнира")
