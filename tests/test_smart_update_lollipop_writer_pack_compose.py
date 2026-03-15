from artifacts.codex import smart_update_lollipop_writer_pack_compose_family_v2_16_2_iter1_2026_03_10 as compose


def test_compose_standard_section_keeps_lead_prefix_for_program_list() -> None:
    block = {"role": "lead", "style": "narrative", "heading": None, "fact_refs": ["PL01"]}
    catalog = {
        "PL01": {
            "fact_id": "PL01",
            "text": "В программе вечера: экскурсия, мастер-классы, викторина.",
            "literal_items": ["экскурсия", "мастер-классы", "викторина"],
            "weight": "high",
            "bucket": "program_list",
        }
    }

    section = compose._compose_standard_section(block, catalog)

    assert section["literal_items"] == ["экскурсия", "мастер-классы", "викторина"]
    assert section["literal_item_source_fact_ids"] == ["PL01"]
    assert section["facts"] == [{"fact_id": "PL01", "text": "В программе вечера.", "priority": 3}]
    assert section["coverage_plan"] == [{"fact_id": "PL01", "mode": "narrative_plus_literal_list"}]


def test_compose_program_section_absorbs_duplicate_list_fact() -> None:
    block = {"role": "program", "style": "list", "heading": "Репертуар", "fact_refs": ["PL01", "PL02"]}
    catalog = {
        "PL01": {
            "fact_id": "PL01",
            "text": "В репертуаре вечеринки песни: «Ветер перемен», «Последняя поэма».",
            "literal_items": ["«Ветер перемен»", "«Последняя поэма»"],
            "weight": "high",
            "bucket": "program_list",
        },
        "PL02": {
            "fact_id": "PL02",
            "text": "На вечеринке будут исполнены песни: «Ветер перемен», «Последняя поэма».",
            "literal_items": [],
            "weight": "medium",
            "bucket": "program_list",
        },
    }

    section = compose._compose_program_section(block, catalog)

    assert section["literal_items"] == ["«Ветер перемен»", "«Последняя поэма»"]
    assert section["facts"] == []
    assert section["coverage_plan"] == [
        {"fact_id": "PL01", "mode": "literal_list"},
        {"fact_id": "PL02", "mode": "absorbed_by_list"},
    ]


def test_compose_program_section_marks_partial_literal_list() -> None:
    block = {"role": "program", "style": "list", "heading": None, "fact_refs": ["PL01"]}
    catalog = {
        "PL01": {
            "fact_id": "PL01",
            "text": "В программе концерта: «Верни мне музыку», «Лучший город земли» и другие композиции.",
            "literal_items": ["«Верни мне музыку»", "«Лучший город земли»"],
            "weight": "high",
            "bucket": "program_list",
        }
    }

    section = compose._compose_program_section(block, catalog)

    assert section["literal_list_is_partial"] is True


def test_canonical_infoblock_label_and_value_are_stable() -> None:
    assert compose._canonical_label("Бесплатно") == "Цена"
    assert compose._canonical_value("Бесплатно", "Цена") == "Бесплатно"
    assert compose._canonical_label("Доступна оплата по «Пушкинской карте».") == "Билеты"
    assert compose._canonical_label("Лекция пройдет в филиале Третьяковской галереи.") == "Локация"


def test_compose_writer_pack_carries_event_type_to_final_writer() -> None:
    layout_result = {
        "event_type": "кинопоказ",
        "layout_result": {
            "precompute": {
                "all_fact_ids": ["SC01", "LG01"],
                "logistics_ids": ["LG01"],
                "title_is_bare": True,
            },
            "payload": {
                "title_strategy": "keep",
                "title_hint_ref": None,
                "blocks": [
                    {"role": "lead", "style": "narrative", "heading": None, "fact_refs": ["SC01"]},
                    {"role": "infoblock", "style": "structured", "heading": None, "fact_refs": ["LG01"]},
                ],
            },
        },
    }
    prioritize_result = {
        "weight_result": {
            "payload": {
                "event_core": [],
                "program_list": [],
                "people_and_roles": [],
                "forward_looking": [],
                "support_context": [
                    {"fact_id": "SC01", "text": "Фильм получил награды.", "literal_items": [], "weight": "medium", "narrative_policy": "include"}
                ],
                "logistics_infoblock": [
                    {"fact_id": "LG01", "text": "Дата: 2026-03-14", "literal_items": [], "weight": "high", "narrative_policy": "include"}
                ],
                "uncertain": [],
            }
        }
    }

    result = compose._compose_writer_pack(
        event_id=2659,
        title="Посторонний",
        layout_result=layout_result,
        prioritize_result=prioritize_result,
    )

    assert result["payload"]["event_type"] == "кинопоказ"
