from artifacts.codex import smart_update_lollipop_facts_extract_family_v2_16_2_2026_03_09 as extract


def _stage(stage_id: str) -> dict:
    return next(item for item in extract.STAGE_SPECS if item["stage_id"] == stage_id)


def test_build_prompt_adds_exhibition_density_rules_for_card_stage() -> None:
    prompt = extract._build_prompt(
        stage=_stage("facts.extract_card.v1"),
        title="Коллекция украшений 1930–1960-х годов",
        event_type="выставка",
        source_excerpt="В витринах будут представлены 240 предметов из музейной коллекции.",
        raw_facts=[
            "На выставке будут представлены 240 предметов из музейной коллекции.",
            "Коллекция считается знаковой для российских музеев.",
        ],
    )

    assert "Для выставок считай curator/history/collection-detail facts релевантными" in prompt
    assert "Для выставки card-facts могут включать название экспозиции, размер коллекции, эпоху" in prompt


def test_build_prompt_adds_exhibition_maker_detail_rules_for_profiles_stage() -> None:
    prompt = extract._build_prompt(
        stage=_stage("facts.extract_profiles.v1"),
        title="Коллекция украшений 1930–1960-х годов",
        event_type="выставка",
        source_excerpt="За каждым шедевром стоит художник-дизайнер с инновационным подходом к созданию бижутерии.",
        raw_facts=[
            "За каждым шедевром стоит художник-дизайнер с инновационным подходом к созданию бижутерии.",
        ],
    )

    assert "Если персон нет, stage всё равно релевантен" in prompt
    assert "Не обрезай lines вроде `за каждым шедевром стоит художник-дизайнер...`" in prompt


def test_build_prompt_for_exhibition_support_forbids_title_based_audience_inference() -> None:
    prompt = extract._build_prompt(
        stage=_stage("facts.extract_support.v1"),
        title="Доступная роскошь",
        event_type="выставка",
        source_excerpt="Билеты. Музей работает 8 и 9 марта с 10:00 до 18:00.",
        raw_facts=[
            "Для посещения выставки требуется приобрести билет.",
            "Музей янтаря работает 8 и 9 марта с 10:00 до 18:00.",
        ],
    )

    assert "Не делай выводы о широкой аудитории, возрасте или доступности" in prompt
    assert "не выводи широкую аудиторию, возрастные ограничения" in prompt


def test_build_prompt_for_exhibition_performer_forbids_bare_subject_name() -> None:
    prompt = extract._build_prompt(
        stage=_stage("facts.extract_performer.v1"),
        title="Выставка «Королева Луиза: идеал или красивая легенда?»",
        event_type="выставка",
        source_excerpt="10 марта откроют выставку «Королева Луиза: идеал или красивая легенда?».",
        raw_facts=[
            "Выставка посвящена биографии и личности королевы Луизы.",
        ],
    )

    assert "Имя без роли, статуса, attribution или credibility-line не возвращай" in prompt
    assert "Для выставок performer-stage обычно пустой" in prompt


def test_build_prompt_keeps_non_exhibition_prompt_clean() -> None:
    prompt = extract._build_prompt(
        stage=_stage("facts.extract_card.v1"),
        title="Посторонний",
        event_type="кинопоказ",
        source_excerpt="Режиссёр: Франсуа Озон.",
        raw_facts=["Фильм «Посторонний» — экранизация романа Альбера Камю."],
    )

    assert "Для выставок считай curator/history/collection-detail facts релевантными" not in prompt
    assert "Для выставки card-facts могут включать название экспозиции" not in prompt


def test_build_run_artifact_paths_supports_custom_run_label() -> None:
    run_slug, trace_root, out_json_path, out_report_path = extract._build_run_artifact_paths(
        "2026-03-12a",
        "v2_16_2_iter9_batch01",
    )

    assert run_slug == "smart_update_lollipop_facts_extract_family_v2_16_2_iter9_batch01_2026-03-12a"
    assert str(trace_root).endswith(run_slug)
    assert str(out_json_path).endswith(f"{run_slug}.json")
    assert str(out_report_path).endswith("smart-update-gemma-event-copy-v2-16-2-lollipop-family-facts-extract-lab-iter9-2026-03-12a.md")
