from artifacts.codex import smart_update_lollipop_editorial_layout_family_v2_16_2_iter1_2026_03_10 as layout


def test_prioritized_fact_pack_excludes_suppressed_narrative_facts() -> None:
    weighted_pack = {
        "event_core": [
            {
                "fact_id": "EC01",
                "text": "Главный факт события.",
                "weight": "high",
                "literal_items": [],
                "narrative_policy": "include",
            }
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [],
        "logistics_infoblock": [
            {
                "fact_id": "LG01",
                "text": "Дата: 2026-03-14",
                "weight": "high",
                "literal_items": [],
                "narrative_policy": "include",
            }
        ],
        "support_context": [
            {
                "fact_id": "SC01",
                "text": "В афише перечислены другие постановки.",
                "weight": "medium",
                "literal_items": [],
                "narrative_policy": "suppress",
            }
        ],
        "uncertain": [],
    }

    pack = layout._prioritized_fact_pack(weighted_pack)

    assert [item["fact_id"] for item in pack["event_core"]] == ["EC01"]
    assert pack["support_context"] == []
    assert [item["fact_id"] for item in pack["logistics_infoblock"]] == ["LG01"]


def test_precompute_layout_state_allows_semantic_headings_for_opaque_screening() -> None:
    pack = {
        "event_core": [],
        "program_list": [],
        "people_and_roles": [
            {"fact_id": "PR01", "text": "Режиссёр фильма — Франсуа Озон.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "people_and_roles"}
        ],
        "forward_looking": [],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "Фильм «Посторонний» является экранизацией одноимённого романа Альбера Камю.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Премьера фильма состоялась на Венецианском кинофестивале в 2025 году.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC03", "text": "Действие фильма разворачивается в Алжире в 1938 году.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }

    precompute = layout._precompute_layout_state(
        event_type="кинопоказ",
        pack=pack,
        lead_payload={"event_title": "Посторонний", "lead_fact_id": "SC01", "lead_support_id": "PR01"},
    )

    assert precompute["allow_semantic_headings"] is True
    assert precompute["title_needs_format_anchor"] is True


def test_precompute_layout_state_recommends_multi_body_split_for_mixed_dense_case() -> None:
    pack = {
        "event_core": [
            {"fact_id": "EC01", "text": "Выставка посвящена украшениям 1930-1960-х годов.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC02", "text": "Экспозиция показывает, как менялся дизайн украшений в середине века.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC03", "text": "В коллекции представлены броши, колье и серьги разных эпох.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC04", "text": "Среди экспонатов есть редкие авторские украшения.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "На моду этого периода повлияли Великая депрессия и Вторая мировая война.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "В экспозиции можно увидеть, как исторические события меняли декоративный язык украшений.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }

    precompute = layout._precompute_layout_state(
        event_type="выставка",
        pack=pack,
        lead_payload={"event_title": "Коллекция украшений 1930–1960-х годов", "lead_fact_id": "EC01", "lead_support_id": "EC02"},
    )

    assert precompute["multi_body_split_recommended"] is True
    assert precompute["body_block_floor"] == 2
    assert precompute["body_cluster_count"] == 2


def test_precompute_layout_state_treats_forward_looking_as_separate_dense_cluster() -> None:
    pack = {
        "event_core": [
            {"fact_id": "EC01", "text": "Проект задуман как социальная сеть для креативных индустрий.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC02", "text": "Платформа поможет искать единомышленников и запускать совместные проекты.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
        ],
        "program_list": [
            {"fact_id": "PL01", "text": "В программе выступления артистов и чтение стихов.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "program_list"},
        ],
        "people_and_roles": [],
        "forward_looking": [
            {"fact_id": "FL01", "text": "На презентации расскажут о задачах и устройстве платформы.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "forward_looking"},
            {"fact_id": "FL02", "text": "На презентации обсудят, зачем появился проект и какую проблему он решает.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "forward_looking"},
            {"fact_id": "FL03", "text": "Откроется предрегистрация для участников проекта.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "forward_looking"},
        ],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [],
        "uncertain": [],
    }

    precompute = layout._precompute_layout_state(
        event_type="презентация",
        pack=pack,
        lead_payload={"event_title": "Собакусъел", "lead_fact_id": "FL01", "lead_support_id": "PL01"},
    )

    assert precompute["multi_body_split_recommended"] is True
    assert precompute["body_block_floor"] == 2
    assert precompute["body_cluster_count"] == 2


def test_clean_layout_plan_does_not_inject_body_heading_for_opaque_screening() -> None:
    pack = {
        "event_core": [],
        "program_list": [],
        "people_and_roles": [
            {"fact_id": "PR01", "text": "Режиссёр фильма — Франсуа Озон.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "people_and_roles"}
        ],
        "forward_looking": [],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "Фильм «Посторонний» является экранизацией одноимённого романа Альбера Камю.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Премьера фильма состоялась на Венецианском кинофестивале в 2025 году.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC03", "text": "Действие фильма разворачивается в Алжире в 1938 году.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }
    lead_payload = {"event_title": "Посторонний", "lead_fact_id": "SC01", "lead_support_id": "PR01"}
    precompute = layout._precompute_layout_state(event_type="кинопоказ", pack=pack, lead_payload=lead_payload)

    cleaned = layout._clean_layout_plan(
        {
            "title_strategy": "keep",
            "title_hint_ref": None,
            "blocks": [
                {"role": "lead", "fact_refs": ["SC01", "PR01"], "style": "narrative", "heading": None},
                {"role": "body", "fact_refs": ["SC02", "SC03"], "style": "narrative", "heading": None},
                {"role": "infoblock", "fact_refs": ["LG01"], "style": "structured", "heading": None},
            ],
        },
        title="Посторонний",
        pack=pack,
        lead_payload=lead_payload,
        precompute=precompute,
    )

    body_block = next(block for block in cleaned["blocks"] if block["role"] == "body")

    assert body_block["heading"] is None


def test_clean_layout_plan_splits_dense_single_body_block_at_cluster_boundary() -> None:
    pack = {
        "event_core": [
            {"fact_id": "EC01", "text": "Выставка посвящена украшениям 1930-1960-х годов.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC02", "text": "Экспозиция показывает, как менялся дизайн украшений в середине века.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC03", "text": "В коллекции представлены броши, колье и серьги разных эпох.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
            {"fact_id": "EC04", "text": "Среди экспонатов есть редкие авторские украшения.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"},
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "На моду этого периода повлияли Великая депрессия и Вторая мировая война.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "В экспозиции можно увидеть, как исторические события меняли декоративный язык украшений.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }
    lead_payload = {"event_title": "Коллекция украшений 1930–1960-х годов", "lead_fact_id": "EC01", "lead_support_id": "EC02"}
    precompute = layout._precompute_layout_state(event_type="выставка", pack=pack, lead_payload=lead_payload)

    cleaned = layout._clean_layout_plan(
        {
            "title_strategy": "keep",
            "title_hint_ref": None,
            "blocks": [
                {"role": "lead", "fact_refs": ["EC01", "EC02"], "style": "narrative", "heading": None},
                {"role": "body", "fact_refs": ["EC03", "EC04", "SC01", "SC02"], "style": "narrative", "heading": "О коллекции"},
                {"role": "infoblock", "fact_refs": ["LG01"], "style": "structured", "heading": None},
            ],
        },
        title="Коллекция украшений 1930–1960-х годов",
        pack=pack,
        lead_payload=lead_payload,
        precompute=precompute,
    )

    body_blocks = [block for block in cleaned["blocks"] if block["role"] == "body"]

    assert len(body_blocks) == 2
    assert body_blocks[0]["heading"] == "О коллекции"
    assert body_blocks[0]["fact_refs"] == ["EC03", "EC04"]
    assert body_blocks[1]["heading"] is None
    assert body_blocks[1]["fact_refs"] == ["SC01", "SC02"]
    assert cleaned["cleaning_stats"]["body_split_floor_applied"] is True


def test_clean_layout_plan_does_not_split_single_cluster_body_block() -> None:
    pack = {
        "event_core": [],
        "program_list": [
            {"fact_id": "PL01", "text": "В программе вечера выступления и танцы.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "program_list"}
        ],
        "people_and_roles": [],
        "forward_looking": [],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "Вечер вдохновлен русскими традициями.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Программа объединяет музыку, танец и бытовую культуру.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC03", "text": "Участники увидят народные образы в сценическом формате.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC04", "text": "Вечер обращается к русскому фольклорному наследию.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }
    lead_payload = {"event_title": "Вечер в русском стиле", "lead_fact_id": "PL01", "lead_support_id": ""}
    precompute = layout._precompute_layout_state(event_type="вечер", pack=pack, lead_payload=lead_payload)

    cleaned = layout._clean_layout_plan(
        {
            "title_strategy": "keep",
            "title_hint_ref": None,
            "blocks": [
                {"role": "lead", "fact_refs": ["PL01"], "style": "narrative", "heading": None},
                {"role": "body", "fact_refs": ["SC01", "SC02", "SC03", "SC04"], "style": "narrative", "heading": "О вечере"},
                {"role": "infoblock", "fact_refs": ["LG01"], "style": "structured", "heading": None},
            ],
        },
        title="Вечер в русском стиле",
        pack=pack,
        lead_payload=lead_payload,
        precompute=precompute,
    )

    body_blocks = [block for block in cleaned["blocks"] if block["role"] == "body"]

    assert precompute["multi_body_split_recommended"] is False
    assert len(body_blocks) == 1
    assert cleaned["cleaning_stats"]["body_split_floor_applied"] is False


def test_build_layout_prompt_mentions_heading_guardrail_and_event_type_examples() -> None:
    input_payload = {
        "event_id": 2747,
        "title": "Последнее метро",
        "event_type": "кинопоказ",
        "source_count": 1,
        "source_excerpt": "Краткое описание",
        "raw_facts": [],
        "lead_fact_id": "SC01",
        "lead_support_id": "PR01",
        "density": "standard",
        "has_long_program": False,
        "non_logistics_total": 5,
        "body_cluster_count": 2,
        "body_block_floor": 2,
        "multi_body_split_recommended": True,
        "title_is_bare": True,
        "title_needs_format_anchor": True,
        "allow_semantic_headings": True,
        "heading_guardrail_recommended": True,
        "all_fact_ids": ["SC01", "PR01", "SC02", "SC03", "LG01"],
        "prioritized_facts": {
            "event_core": [],
            "program_list": [],
            "people_and_roles": [],
            "forward_looking": [],
            "logistics_infoblock": [],
            "support_context": [],
            "uncertain": [],
        },
    }

    prompt = layout._build_layout_prompt(input_payload=input_payload)

    assert "heading_guardrail_recommended" in prompt
    assert "body_block_floor" in prompt
    assert "multi_body_split_recommended" in prompt
    assert "Use `event_type` to choose heading vocabulary" in prompt
    assert "EXAMPLE: screening" in prompt
    assert "EXAMPLE: lecture" in prompt


def test_audit_layout_flags_dense_case_without_headings() -> None:
    pack = {
        "event_core": [
            {"fact_id": "EC01", "text": "Лекция посвящена нескольким художницам.", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "event_core"}
        ],
        "program_list": [],
        "people_and_roles": [
            {"fact_id": "PR01", "text": "Будут рассмотрены работы Веры Мухиной.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "people_and_roles"}
        ],
        "forward_looking": [
            {"fact_id": "FL01", "text": "После лекции пройдет разговор с куратором.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "forward_looking"}
        ],
        "logistics_infoblock": [
            {"fact_id": "LG01", "text": "Дата: 2026-03-14", "weight": "high", "literal_items": [], "narrative_policy": "include", "bucket": "logistics_infoblock"}
        ],
        "support_context": [
            {"fact_id": "SC01", "text": "Разговор затронет опыт авангардных школ.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Отдельный блок будет посвящен скульптуре.", "weight": "medium", "literal_items": [], "narrative_policy": "include", "bucket": "support_context"},
        ],
        "uncertain": [],
    }
    lead_payload = {"event_title": "Женщины авангарда", "lead_fact_id": "EC01", "lead_support_id": "PR01"}
    precompute = layout._precompute_layout_state(event_type="лекция", pack=pack, lead_payload=lead_payload)
    plan_payload = {
        "title_strategy": "keep",
        "title_hint_ref": None,
        "blocks": [
            {"role": "lead", "fact_refs": ["EC01", "PR01"], "style": "narrative", "heading": None},
            {"role": "body", "fact_refs": ["FL01", "SC01", "SC02"], "style": "narrative", "heading": None},
            {"role": "infoblock", "fact_refs": ["LG01"], "style": "structured", "heading": None},
        ],
    }

    audit = layout._audit_layout(
        plan_payload=plan_payload,
        pack=pack,
        precompute=precompute,
        lead_payload=lead_payload,
        title="Женщины авангарда",
    )

    assert precompute["heading_guardrail_recommended"] is True
    assert "missing_headings_for_dense_case" in audit["flags"]
    assert audit["metrics"]["heading_guardrail_recommended"] is True
