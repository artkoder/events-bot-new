from smart_event_update import _append_missing_small_list, _normalize_bullet_markers


def test_append_missing_small_list_adds_markdown_list() -> None:
    source_text = """
Дорогой мечтатель!

· Поплаваешь в компании величественных китов.
· Коснешься таинственного морского дна.
· Позволишь волнам смыть напряжение.
"""
    source_text = _normalize_bullet_markers(source_text) or source_text
    desc = "Короткий рерайт без списка."
    out = _append_missing_small_list(description=desc, source_text=source_text, source_type="telegram")
    assert out is not None
    assert "### Что вас ждёт" in out
    assert "- Поплаваешь в компании величественных китов." in out
    assert "- Коснешься таинственного морского дна." in out
    assert "- Позволишь волнам смыть напряжение." in out


def test_append_missing_small_list_does_not_duplicate_existing_list() -> None:
    source_text = "- Пункт 1\n- Пункт 2"
    desc = "### Подробности\n- Пункт 1\n- Пункт 2"
    out = _append_missing_small_list(description=desc, source_text=source_text, source_type="telegram")
    assert out == desc

