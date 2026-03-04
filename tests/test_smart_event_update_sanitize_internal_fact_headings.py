from __future__ import annotations


def test_sanitize_description_drops_internal_fact_headings_only() -> None:
    from smart_event_update import _sanitize_description_output

    text = (
        "Вступительный абзац.\n\n"
        "Факты для лога источников:\n\n"
        "Формат: лекция\n"
        "Спикер: Максим\n\n"
        "Завершающий абзац."
    )
    out = _sanitize_description_output(text, source_text="") or ""

    assert "Факты для лога источников" not in out
    assert "Вступительный абзац." in out
    assert "Формат: лекция" in out
    assert "Завершающий абзац." in out


def test_sanitize_description_drops_internal_log_block_when_heading_and_content_in_same_paragraph() -> None:
    from smart_event_update import _sanitize_description_output

    text = (
        "Лид.\n\n"
        "Факты для лога источников:\n"
        "Формат: лекция\n"
        "Спикер: Максим\n\n"
        "Финал."
    )
    out = _sanitize_description_output(text, source_text="") or ""

    assert "Факты для лога источников" not in out
    assert "Формат: лекция" not in out
    assert "Лид." in out
    assert "Финал." in out


def test_sanitize_description_drops_facts_about_event_heading_only() -> None:
    from smart_event_update import _sanitize_description_output

    text = "Лид.\n\nФакты о событии:\n\nФормат: игра."
    out = _sanitize_description_output(text, source_text="") or ""
    assert "Факты о событии" not in out
    assert "Формат: игра." in out


def test_sanitize_description_strips_facts_heading_line_but_keeps_content() -> None:
    from smart_event_update import _sanitize_description_output

    text = (
        "Абзац.\n\n"
        "Facts:\n"
        "Мероприятие платное.\n"
        "Встреча проходит в школе.\n\n"
        "Хвост."
    )
    out = _sanitize_description_output(text, source_text="") or ""

    assert "Facts:" not in out
    assert "Мероприятие платное." in out
    assert "Встреча проходит в школе." in out


def test_sanitize_description_demotes_overlong_heading_and_strips_inline_facts_prefix() -> None:
    from smart_event_update import _sanitize_description_output

    text = (
        "### Масленичные гуляния Калининградский янтарный комбинат приглашает жителей и гостей на празднование Масленицы. "
        "Гостей ждут хороводы и песни. Facts: Участие бесплатное.\n\n"
        "Хвост."
    )
    out = _sanitize_description_output(text, source_text="") or ""
    assert "### Масленичные гуляния" not in out
    assert "Facts:" not in out
    assert "Участие бесплатное." in out
    assert "Хвост." in out


def test_sanitize_description_demotes_long_heading_without_punctuation() -> None:
    from smart_event_update import _sanitize_description_output

    text = (
        "### Масленичные гуляния Калининградский янтарный комбинат приглашает жителей и гостей\n\n"
        "Хвост."
    )
    out = _sanitize_description_output(text, source_text="") or ""
    assert "### Масленичные гуляния" not in out
    assert "Калининградский янтарный комбинат приглашает жителей и гостей" in out
    assert "Хвост." in out


def test_sanitize_description_strips_bold_facts_prefix_inline() -> None:
    from smart_event_update import _sanitize_description_output

    text = "Абзац. **Facts:** Мероприятие платное.\n\nХвост."
    out = _sanitize_description_output(text, source_text="") or ""

    assert "**Facts:**" not in out
    assert "Facts:" not in out
    assert "Мероприятие платное." in out
    assert "Хвост." in out
