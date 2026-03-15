from artifacts.codex import smart_update_lollipop_facts_prioritize_family_v2_16_2_iter1_2026_03_10 as prioritize


def test_augment_fact_pack_from_raw_facts_recovers_historical_context() -> None:
    fact_pack = {
        "stage_notes": "",
        "event_core": [
            {"fact_id": "EC01", "text": "Выставка рассказывает об истории зарубежного ювелирного и бижутерийного искусства.", "literal_items": [], "evidence_record_ids": []}
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [],
        "logistics_infoblock": [],
        "support_context": [],
        "uncertain": [],
    }

    augmented = prioritize._augment_fact_pack_from_raw_facts(
        fact_pack,
        event_type="выставка",
        raw_facts=[
            "Коллекция охватывает период с 1930 по 1960 год.",
            "Экспозиция отражает влияние Великой депрессии и Второй мировой войны на моду.",
        ],
    )

    rescued_texts = [item["text"] for item in augmented["support_context"]]

    assert "Экспозиция отражает влияние Великой депрессии и Второй мировой войны на моду." in rescued_texts
    assert augmented["rescue_stats"]["rescued_fact_ids"] == ["SC01"]


def test_apply_narrative_policies_suppresses_cross_promo_and_fillers() -> None:
    weighted_pack = {
        "stage_notes": "",
        "event_core": [
            {"fact_id": "EC01", "text": "Основной факт.", "literal_items": [], "evidence_record_ids": [], "weight": "high", "weight_reasoning": "", "bucket": "event_core"},
            {"fact_id": "EC02", "text": "Еще один основной факт.", "literal_items": [], "evidence_record_ids": [], "weight": "high", "weight_reasoning": "", "bucket": "event_core"},
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [
            {"fact_id": "FL01", "text": "Будет показ специальной версии.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "forward_looking"},
            {"fact_id": "FL02", "text": "После показа пройдет обсуждение.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "forward_looking"},
        ],
        "logistics_infoblock": [],
        "support_context": [
            {
                "fact_id": "SC01",
                "text": "В афише указаны даты ближайших спектаклей на средней сцене: 04.03 - Коралина, 10.03 - Нюрнберг, 17.03 - Материнское поле.",
                "literal_items": [],
                "evidence_record_ids": [],
                "weight": "medium",
                "weight_reasoning": "",
                "bucket": "support_context",
            },
            {
                "fact_id": "SC02",
                "text": "Выставка предназначена для широкой аудитории, без возрастных ограничений.",
                "literal_items": [],
                "evidence_record_ids": [],
                "weight": "low",
                "weight_reasoning": "",
                "bucket": "support_context",
            },
        ],
        "uncertain": [],
        "cleaning_stats": {"auto_added_count": 0, "rescued_fact_ids": []},
    }

    prioritized = prioritize._apply_narrative_policies(weighted_pack)
    policy_map = {
        item["fact_id"]: item["narrative_policy"]
        for item in prioritize._flat_facts(prioritized)
    }

    assert policy_map["SC01"] == "suppress"
    assert policy_map["SC02"] == "suppress"
    assert prioritized["policy_stats"]["suppressed_count"] == 2


def test_apply_narrative_policies_suppresses_hospitality_and_generic_audience_pitch() -> None:
    weighted_pack = {
        "stage_notes": "",
        "event_core": [
            {"fact_id": "EC01", "text": "На презентации расскажут о задачах и возможностях платформы.", "literal_items": [], "evidence_record_ids": [], "weight": "high", "weight_reasoning": "", "bucket": "event_core"},
            {"fact_id": "EC02", "text": "Проект задуман как пространство для поиска единомышленников.", "literal_items": [], "evidence_record_ids": [], "weight": "high", "weight_reasoning": "", "bucket": "event_core"},
            {"fact_id": "EC03", "text": "Мероприятие будет интересно представителям креативной среды и тем, кто ищет новые возможности для сотрудничества.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "event_core"},
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [
            {"fact_id": "FL01", "text": "Откроется предрегистрация для участников проекта.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "forward_looking"},
        ],
        "logistics_infoblock": [],
        "support_context": [
            {"fact_id": "SC01", "text": "Посетителям рекомендуется принести с собой печенье.", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Чай будет предоставлен организаторами.", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
        ],
        "uncertain": [],
        "cleaning_stats": {"auto_added_count": 0, "rescued_fact_ids": []},
    }

    prioritized = prioritize._apply_narrative_policies(weighted_pack, event_type="presentation")
    policy_map = {
        item["fact_id"]: item["narrative_policy"]
        for item in prioritize._flat_facts(prioritized)
    }

    assert policy_map["EC03"] == "suppress"
    assert policy_map["SC01"] == "suppress"
    assert policy_map["SC02"] == "suppress"
    assert prioritized["policy_stats"]["suppressed_count"] == 3


def test_apply_narrative_policies_promotes_screening_support_when_no_event_core_exists() -> None:
    weighted_pack = {
        "stage_notes": "",
        "event_core": [],
        "program_list": [],
        "people_and_roles": [
            {"fact_id": "PR01", "text": "Режиссёр фильма — Франсуа Трюффо.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "people_and_roles"},
        ],
        "forward_looking": [],
        "logistics_infoblock": [],
        "support_context": [
            {"fact_id": "SC01", "text": "Фильм «Последнее метро» получил 10 премий «Сезар».", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
            {"fact_id": "SC02", "text": "Действие фильма разворачивается в оккупированном Париже.", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
            {"fact_id": "SC03", "text": "Фильм рассказывает о театре, работающем под контролем нацистов.", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
            {"fact_id": "SC04", "text": "В центре сюжета — любовный треугольник.", "literal_items": [], "evidence_record_ids": [], "weight": "low", "weight_reasoning": "", "bucket": "support_context"},
        ],
        "uncertain": [],
        "cleaning_stats": {"auto_added_count": 0, "rescued_fact_ids": []},
    }

    prioritized = prioritize._apply_narrative_policies(weighted_pack, event_type="кинопоказ")
    weight_map = {
        item["fact_id"]: item["weight"]
        for item in prioritize._flat_facts(prioritized)
    }

    assert weight_map["SC02"] == "medium"
    assert weight_map["SC03"] == "medium"
    assert weight_map["SC04"] == "medium"
    assert prioritized["policy_stats"]["promoted_count"] == 3


def test_clean_lead_prefers_forward_looking_anchor_for_opaque_presentation_title() -> None:
    weighted_pack = {
        "stage_notes": "",
        "event_core": [
            {"fact_id": "EC01", "text": "Проект «Собакусъел» представляет собой социальную сеть для профессионалов креативных индустрий.", "literal_items": [], "evidence_record_ids": [], "weight": "high", "weight_reasoning": "", "bucket": "event_core", "narrative_policy": "include"},
        ],
        "program_list": [],
        "people_and_roles": [],
        "forward_looking": [
            {"fact_id": "FL01", "text": "На презентации расскажут о задачах, устройстве и возможностях платформы.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "forward_looking", "narrative_policy": "include"},
        ],
        "logistics_infoblock": [],
        "support_context": [],
        "uncertain": [],
        "cleaning_stats": {"auto_added_count": 0, "rescued_fact_ids": []},
    }

    payload = prioritize._clean_lead(
        {"lead_fact_id": "EC01", "lead_support_id": "", "reasoning": "", "stage_notes": ""},
        weighted_pack,
        title="Собакусъел",
        event_type="presentation",
    )

    assert payload["lead_fact_id"] == "FL01"
    assert payload["lead_support_id"] == "EC01"
    assert "lead_missing_format_anchor" in payload["cleaning_stats"]["fallback_reasons"]


def test_clean_lead_prefers_non_people_fact_for_opaque_screening_title() -> None:
    weighted_pack = {
        "stage_notes": "",
        "event_core": [],
        "program_list": [],
        "people_and_roles": [
            {"fact_id": "PR01", "text": "Режиссёр фильма — Франсуа Озон.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "people_and_roles", "narrative_policy": "include"},
        ],
        "forward_looking": [],
        "logistics_infoblock": [],
        "support_context": [
            {"fact_id": "SC01", "text": "Фильм «Посторонний» является экранизацией одноимённого романа Альбера Камю.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "support_context", "narrative_policy": "include"},
            {"fact_id": "SC02", "text": "Премьера фильма состоялась на Венецианском кинофестивале в 2025 году.", "literal_items": [], "evidence_record_ids": [], "weight": "medium", "weight_reasoning": "", "bucket": "support_context", "narrative_policy": "include"},
        ],
        "uncertain": [],
        "cleaning_stats": {"auto_added_count": 0, "rescued_fact_ids": []},
    }

    payload = prioritize._clean_lead(
        {"lead_fact_id": "PR01", "lead_support_id": "", "reasoning": "", "stage_notes": ""},
        weighted_pack,
        title="Посторонний",
        event_type="кинопоказ",
    )

    assert payload["lead_fact_id"] == "SC01"
    assert payload["lead_support_id"] == "PR01"
    assert "lead_missing_format_anchor" in payload["cleaning_stats"]["fallback_reasons"]


def test_build_lead_prompt_includes_format_anchor_examples_for_opaque_titles() -> None:
    prompt = prioritize._build_lead_prompt(
        event_id=2659,
        title="Посторонний",
        event_type="кинопоказ",
        weighted_pack={
            "event_core": [],
            "program_list": [],
            "people_and_roles": [],
            "forward_looking": [],
            "logistics_infoblock": [],
            "support_context": [],
            "uncertain": [],
        },
    )

    assert "title_needs_format_anchor" in prompt
    assert "WRONG lead" in prompt
    assert "Режиссёр фильма" in prompt
    assert "what kind of attendable event is this" in prompt
