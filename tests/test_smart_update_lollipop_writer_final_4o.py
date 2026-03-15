from artifacts.codex import smart_update_lollipop_writer_final_4o_family_v2_16_2_iter1_2026_03_10 as writer_final


def _base_pack(*, strategy: str = "keep") -> dict:
    return {
        "event_type": "лекция",
        "title_context": {
            "original_title": "Исходный заголовок",
            "strategy": strategy,
            "hint_fact_id": "EC01" if strategy == "enhance" else None,
            "hint_fact_text": "Лекция о художницах" if strategy == "enhance" else None,
            "is_bare": strategy == "enhance",
        },
        "sections": [
            {
                "role": "lead",
                "style": "narrative",
                "heading": None,
                "fact_ids": ["EC01"],
                "facts": [{"fact_id": "EC01", "text": "Событие посвящено художницам.", "priority": 3}],
                "coverage_plan": [{"fact_id": "EC01", "mode": "narrative"}],
            },
            {
                "role": "program",
                "style": "list",
                "heading": "Программа",
                "fact_ids": ["PL01"],
                "facts": [],
                "coverage_plan": [{"fact_id": "PL01", "mode": "literal_list"}],
                "literal_items": ["«Первый пункт»", "«Второй пункт»"],
                "literal_list_is_partial": False,
            },
        ],
        "infoblock": [
            {"fact_id": "LG01", "label": "Дата", "value": "2026-03-14"},
            {"fact_id": "LG02", "label": "Время", "value": "16:00"},
            {"fact_id": "LG03", "label": "Цена", "value": "500 ₽"},
        ],
        "constraints": {
            "must_cover_fact_ids": ["EC01", "PL01"],
            "infoblock_fact_ids": ["LG01", "LG02", "LG03"],
            "headings": ["Программа"],
            "list_required": True,
            "no_logistics_in_narrative": True,
        },
    }


def test_apply_writer_output_keeps_original_title_on_keep_strategy() -> None:
    pack = _base_pack()

    applied = writer_final._apply_writer_output(
        pack,
        {"title": "Переписанный заголовок", "description_md": "Короткое описание."},
    )

    assert applied["title"] == "Исходный заголовок"
    assert applied["description_md"] == "Короткое описание."


def test_validate_writer_output_flags_infoblock_leak_and_missing_literal_items() -> None:
    pack = _base_pack()

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Другой заголовок",
            "description_md": "Событие посвящено художницам. Начало в 16:00.\n\n### Программа\n- «Первый пункт»",
        },
    )

    assert "title.keep_overridden_by_model" in validation.warnings
    assert any(error.startswith("infoblock.") and ":LG02" in error for error in validation.errors)
    assert "literal.missing:«Второй пункт»" in validation.errors


def test_validate_writer_output_warns_on_literal_format_mutation_without_failing() -> None:
    pack = _base_pack()

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": "Событие посвящено художницам.\n\n### Программа\n- Первый пункт\n- Второй пункт",
        },
    )

    assert validation.errors == []
    assert "literal.format_mutation:«Первый пункт»" in validation.warnings
    assert "literal.format_mutation:«Второй пункт»" in validation.warnings


def test_validate_writer_output_requires_actual_enhanced_title() -> None:
    pack = _base_pack(strategy="enhance")

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": "Событие посвящено художницам.\n\n### Программа\n- «Первый пункт»\n- «Второй пункт»",
        },
    )

    assert "title.enhance_unchanged" in validation.errors


def test_validate_writer_output_does_not_count_initials_as_extra_lead_sentences() -> None:
    pack = _base_pack()

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": (
                "Участники узнают об истории создания первой карты области. "
                "К. М. Кишкин — художник, чьё творчество будет представлено на мастер-классе.\n\n"
                "### Программа\n- «Первый пункт»\n- «Второй пункт»"
            ),
        },
    )

    assert not any(item.startswith("lead.too_long") for item in validation.warnings)


def test_validate_writer_output_requires_intro_before_bullet_list() -> None:
    pack = _base_pack()

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": "Событие посвящено художницам.\n\n- «Первый пункт»\n- «Второй пункт»",
        },
    )

    assert "list.unintroduced_block:3" in validation.errors


def test_validate_writer_output_allows_sentence_intro_for_narrative_plus_literal_list() -> None:
    pack = _base_pack()
    pack["sections"] = [
        {
            "role": "lead",
            "style": "narrative",
            "heading": None,
            "fact_ids": ["PL01"],
            "facts": [{"fact_id": "PL01", "text": "В программе вечера.", "priority": 3}],
            "coverage_plan": [{"fact_id": "PL01", "mode": "narrative_plus_literal_list"}],
            "literal_items": ["«Первый пункт»", "«Второй пункт»"],
            "literal_list_is_partial": False,
        }
    ]
    pack["constraints"]["must_cover_fact_ids"] = ["PL01"]
    pack["constraints"]["headings"] = []

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": "Это вечер с насыщенной программой.\n\n- «Первый пункт»\n- «Второй пункт»",
        },
    )

    assert validation.errors == []


def test_validate_writer_output_requires_partial_intro_marker() -> None:
    pack = _base_pack()
    pack["sections"][1]["literal_list_is_partial"] = True

    validation = writer_final._validate_writer_output(
        pack,
        {
            "title": "Исходный заголовок",
            "description_md": "Событие посвящено художницам.\n\nПрозвучат композиции:\n- «Первый пункт»\n- «Второй пункт»",
        },
    )

    assert "list.partial_intro_missing" in validation.errors


def test_build_prompt_includes_structure_plan_format_clarity_and_length_guidance() -> None:
    pack = _base_pack(strategy="enhance")
    pack["event_type"] = "кинопоказ"
    pack["title_context"]["original_title"] = "Посторонний"
    pack["title_context"]["hint_fact_text"] = "Кинопоказ фильма по роману Альбера Камю."
    pack["title_context"]["is_bare"] = True
    pack["sections"].append(
        {
            "role": "body",
            "style": "narrative",
            "heading": "О фильме",
            "fact_ids": ["SC01", "SC02", "SC03"],
            "facts": [
                {"fact_id": "SC01", "text": "Фильм снят по роману Альбера Камю.", "priority": 2},
                {"fact_id": "SC02", "text": "Режиссер — Франсуа Озон.", "priority": 2},
                {"fact_id": "SC03", "text": "В центре сюжета — история Мерсо.", "priority": 1},
            ],
            "coverage_plan": [
                {"fact_id": "SC01", "mode": "narrative"},
                {"fact_id": "SC02", "mode": "narrative"},
                {"fact_id": "SC03", "mode": "narrative"},
            ],
        }
    )
    pack["constraints"]["must_cover_fact_ids"] = ["EC01", "PL01", "SC01", "SC02", "SC03"]

    prompt = writer_final._build_prompt(pack)

    assert "СТРУКТУРА (соблюдай порядок и exact headings):" in prompt
    assert "event_type: \"кинопоказ\"" in prompt
    assert "title_needs_format_clarity: true" in prompt
    assert "lead_needs_format_bridge: true" in prompt
    assert "первое предложение должно сразу прояснить формат события" in prompt
    assert "первое предложение обязано прямо назвать формат события через `event_type`" in prompt
    assert "На каждой границе `section` начинай новый абзац" in prompt
    assert "По объёму ориентируйся примерно на 500-900 знаков" in prompt
    assert "Плохое открытие: `Режиссёр фильма — ...`" in prompt
    assert "use exact heading: ### О фильме" in prompt
