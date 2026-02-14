from smart_event_update import (
    EventCandidate,
    _append_missing_fact_sentences,
    _build_fact_seed_text,
    _candidate_has_new_text,
    _dedupe_paragraphs_preserving_formatting,
    _drop_redundant_poster_facts,
    _drop_reported_speech_duplicates,
    _enforce_merge_non_shrinking_description,
    _fallback_merge_description,
    _initial_added_facts,
    _is_low_signal_sentence,
    _pick_new_description_snippet,
    _pick_new_text_snippet,
    _preserve_blockquotes_from_previous_description,
    _split_overlong_first_person_blockquotes,
    _strip_foreign_schedule_headings,
    _strip_foreign_schedule_noise,
    _demote_redundant_anchor_facts,
)
from models import Event


def test_candidate_has_new_text_uses_source_text_when_excerpt_is_stale() -> None:
    event = Event(
        id=1,
        title="Лорд Фаунтлерой",
        description="Спектакль в драмтеатре. Премия Арлекин-2010.",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        source_text="старый текст",
    )
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3802",
        source_text=(
            "Спектакль «Лорд Фаунтлерой».\n"
            "Прекрасный дуэт Александра Егорова и Павла Самоловова."
        ),
        raw_excerpt="Спектакль «Лорд Фаунтлерой».",
        title="Лорд Фаунтлерой",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    assert _candidate_has_new_text(candidate, event) is True


def test_pick_new_text_snippet_prefers_new_sentence() -> None:
    before = "Спектакль «Лорд Фаунтлерой»."
    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://source",
        source_text="Спектакль «Лорд Фаунтлерой». Прекрасный дуэт Александра Егорова и Павла Самоловова.",
        raw_excerpt="Спектакль «Лорд Фаунтлерой».",
        title="Лорд Фаунтлерой",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    snippet = _pick_new_text_snippet(candidate, before)
    assert snippet is not None
    assert "дуэт" in snippet.lower()


def test_pick_new_description_snippet_is_from_after_description() -> None:
    before = "Спектакль «Лорд Фаунтлерой»."
    after = "Спектакль «Лорд Фаунтлерой».\n\nПрекрасный дуэт Александра Егорова и Павла Самоловова."
    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://source",
        source_text="04.02 | Мысли мудрых людей на каждый день",
        raw_excerpt="",
        title="Лорд Фаунтлерой",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    snippet = _pick_new_description_snippet(after, before, candidate=candidate)
    assert snippet is not None
    assert "дуэт" in snippet.lower()


def test_noisy_markdown_source_prefers_salient_duet_fact() -> None:
    before = (
        "Спектакль о юном американце, ставшем наследником графства и встретившемся с черствым дедом в Англии. "
        "Уже более 16 лет с теплыми и радостными чувствами зрители всей семьей приходят на спектакль, "
        "поставленный Народным артистом России Михаилом Салесом и удостоенный Российской Национальной "
        "премии «Арлекин-2010» в четырех номинациях."
    )
    source_text = (
        "**скоро в театре**\n\n"
        "**05.02** | [Лорд Фаунтлерой](https://dramteatr39.ru/spektakli/lord-fauntleroj)\n"
        "__добрая сказка для детей и их родителей\n__\n"
        "Уже более 16 лет с теплыми и радостными чувствами зрители всей семьей приходят на спектакль, "
        "поставленный Народным артистом России Михаилом Салесом и удостоенный Российской Национальной "
        "премии «Арлекин-2010» в четырех номинациях.\n\n"
        "Прекрасный дуэт Александра Егорова и Павла Самоловова."
    )
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3802",
        source_text=source_text,
        raw_excerpt="Лорд Фаунтлерой",
        title="Лорд Фаунтлерой",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
    )

    snippet = _pick_new_text_snippet(candidate, before)
    assert snippet is not None
    assert "дуэт" in snippet.lower()

    merged = _fallback_merge_description(before, candidate, max_sentences=1)
    assert merged is not None
    assert "дуэт" in merged.lower()


def test_drop_redundant_poster_facts_hides_source_url_when_same_as_added() -> None:
    facts = [
        "Афиша в источнике: https://files.catbox.moe/2u5ukh.jpg",
        "Добавлена афиша: https://files.catbox.moe/2u5ukh.jpg",
        "Текст дополнен: Пример",
    ]
    filtered = _drop_redundant_poster_facts(facts)
    assert "Добавлена афиша: https://files.catbox.moe/2u5ukh.jpg" in filtered
    assert "Афиша в источнике: https://files.catbox.moe/2u5ukh.jpg" not in filtered


def test_low_signal_sentence_filters_schedule_headers() -> None:
    assert _is_low_signal_sentence("04.02 | Мысли мудрых людей на каждый день") is True
    assert _is_low_signal_sentence("8/2 — Мёртвые души") is True


def test_merge_non_shrinking_keeps_before_when_llm_collapses_text() -> None:
    before = (
        "Это подробное описание спектакля. " * 40
    ).strip()  # ~1200+ chars
    merged = "Короткий дайджест."
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3791",
        source_text=(
            "Подробное описание спектакля. "
            "Прекрасный дуэт Александра Егорова и Павла Самоловова на сцене."
        ),
        raw_excerpt="",
        title="Мёртвые души",
        date="2026-02-07",
        time="18:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    enforced = _enforce_merge_non_shrinking_description(
        before_description=before,
        merged_description=merged,
        candidate=candidate,
        has_new_text=True,
    )
    assert len(enforced) >= len(before)
    assert "дуэт" in enforced.lower()


def test_merge_non_shrinking_prefers_candidate_when_before_and_merged_too_short() -> None:
    before = "Короткое описание."
    merged = "Короткий дайджест."
    candidate_text = ("Подробное описание события с множеством фактов. " * 60).strip()
    candidate = EventCandidate(
        source_type="site",
        source_url="https://dramteatr39.ru/spektakli/mertvye-dushi",
        source_text=candidate_text,
        raw_excerpt="",
        title="Мёртвые души",
        date="2026-02-07",
        time="18:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    enforced = _enforce_merge_non_shrinking_description(
        before_description=before,
        merged_description=merged,
        candidate=candidate,
        has_new_text=True,
    )
    assert len(enforced) >= len(candidate_text) * 0.9


def test_strip_foreign_schedule_headings_removes_other_dates() -> None:
    text = (
        "04.02 | Мысли мудрых людей на каждый день\n"
        "07.02 | Мёртвые души\n"
        "Описание спектакля.\n"
        "08.02 | Три супруги-совершенства\n"
    )
    cleaned = _strip_foreign_schedule_headings(text, event_date="2026-02-07", end_date=None)
    assert "04.02 | Мысли" not in cleaned
    assert "08.02 | Три супруги" not in cleaned
    assert "07.02 | Мёртвые души" in cleaned


def test_strip_foreign_schedule_noise_removes_foreign_list_sentence() -> None:
    text = (
        "Спектакль по мотивам классики.\n"
        "Спектакль является частью театральной недели, в рамках которой также пройдут спектакли "
        "\"Нюрнберг\", \"Мысли мудрых людей на каждый день\", \"Лорд Фаунтлерой\".\n"
        "Показ состоится 7 февраля в 18:00."
    )
    cleaned = _strip_foreign_schedule_noise(
        text,
        event_date="2026-02-07",
        end_date=None,
        event_title="Мёртвые души",
    )
    assert "мысли мудрых" not in cleaned.lower()
    assert "Показ состоится" in cleaned


def test_fact_seed_preserves_style_terms_like_flamenco() -> None:
    event = Event(
        id=1,
        title="🎭 Три супруги-совершенства",
        description="Короткое описание без жанра.",
        date="2026-02-08",
        time="18:00",
        location_name="Драматический театр",
        source_text=(
            "О спектакле. Комедия - в стиле фламенко: испанские страсти, ложь, измены и разоблачения."
        ),
    )
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3791",
        source_text="08.02 | Три супруги-совершенства",
        raw_excerpt="",
        title="Три супруги-совершенства",
        date="2026-02-08",
        time="18:00",
        location_name="Драматический театр",
        city="Калининград",
    )
    seed = _build_fact_seed_text(event, candidate)
    assert "фламенко" in seed.lower()
    rewritten = "В театре пройдет спектакль. Подробности и билеты по ссылке."
    out = _append_missing_fact_sentences(base=seed, rewritten=rewritten, max_sentences=2)
    assert "фламенко" in out.lower()


def test_append_missing_fact_sentences_can_enforce_general_coverage() -> None:
    base = (
        "Проект создан в рамках фестиваля.\n"
        "Указаны варианты проезда к галерее на общественном транспорте и пешком от различных достопримечательностей."
    )
    rewritten = "Проект создан в рамках фестиваля."
    out = _append_missing_fact_sentences(
        base=base,
        rewritten=rewritten,
        max_sentences=2,
        ensure_coverage=True,
    )
    assert "варианты проезда" in out.lower()


def test_append_missing_fact_sentences_keeps_critical_short_facts() -> None:
    base = (
        "Нескучная французская классика. Настоящий театральный хит, проверенный временем.\n"
        "Спектакль начнется в 19:00.\n"
        "Спектакль происходит на Основной сцене."
    )
    rewritten = "12 февраля состоится спектакль «Фигаро»."
    out = _append_missing_fact_sentences(
        base=base,
        rewritten=rewritten,
        max_sentences=8,
        ensure_coverage=True,
    )
    out_lower = out.lower()
    assert "театральный хит" in out_lower
    assert "19:00" in out
    assert "основной сцен" in out_lower


def test_initial_added_facts_do_not_include_textual_thesis_for_created_event() -> None:
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3782",
        source_text=(
            "12.02 | Фигаро\n"
            "Нескучная французская классика. Настоящий театральный хит, проверенный временем."
        ),
        raw_excerpt="12.02 | Фигаро",
        title="Фигаро",
        date="2026-02-12",
        time="",
        location_name="Драматический театр",
        city="Калининград",
    )
    facts = _initial_added_facts(candidate)
    assert not any(str(f).lower().startswith("тезис:") for f in facts)
    assert any(str(f).lower().startswith("дата:") for f in facts)
    assert any(str(f).lower().startswith("локация:") for f in facts)


def test_merge_preserves_existing_blockquote_and_drops_reported_speech_duplicate() -> None:
    before = (
        "Нескучная французская классика.\n\n"
        "> Мне кажется, что сегодня характер Фигаро требует переосмысления."
    )
    merged = (
        "Нескучная французская классика.\n\n"
        "Режиссёр Егор Равинский отмечает, что сегодня характер Фигаро требует переосмысления."
    )
    preserved = _preserve_blockquotes_from_previous_description(
        before_description=before,
        merged_description=merged,
        event_title="Фигаро",
    )
    assert preserved is not None
    # Now drop duplicates: the paraphrase should be removed in favor of the direct quote.
    cleaned = _drop_reported_speech_duplicates(preserved)
    assert cleaned is not None
    assert "> Мне кажется, что сегодня характер Фигаро требует переосмысления." in cleaned
    assert "отмечает, что сегодня характер фигаро требует переосмысления" not in cleaned.lower()


def test_split_overlong_first_person_blockquotes_keeps_only_first_sentence_in_quote() -> None:
    text = (
        "> Мне кажется, что сегодня характер Фигаро требует переосмысления. "
        "В легендарном спектакле Театра сатиры с Андреем Мироновым Фигаро — благородный бунтарь. "
        "Это уже не прямая речь.\n\n"
        "Следующий абзац."
    )
    out = _split_overlong_first_person_blockquotes(text)
    assert out is not None
    parts = [p.strip() for p in out.split("\n\n") if p.strip()]
    assert parts[0].startswith("> ")
    assert parts[0].count(".") == 1
    assert "легендарном спектакле" in parts[1].lower()
    assert parts[2] == "Следующий абзац."


def test_dedupe_paragraphs_preserving_formatting_drops_duplicates_ignoring_trailing_punct() -> None:
    text = (
        "При этом калининградский «Фигаро» живет и действует в уникальном художественном мире.\n\n"
        "При этом калининградский «Фигаро» живет и действует в уникальном художественном мире\n\n"
        "Другой абзац."
    )
    out = _dedupe_paragraphs_preserving_formatting(text)
    assert out is not None
    parts = [p.strip() for p in out.split("\n\n") if p.strip()]
    assert len(parts) == 2
    assert "калининградский" in parts[0].lower()
    assert parts[1] == "Другой абзац."


def test_source_log_demotes_natural_language_date_and_time_to_duplicates() -> None:
    added = [
        "Спектакль будет показан 12 февраля.",
        "Начало спектакля в 19:00.",
        "Дата: 2026-02-12",
        "Время: 19:00",
    ]
    dup: list[str] = []
    new_added, new_dup = _demote_redundant_anchor_facts(
        added,
        dup,
        event_date="2026-02-12",
        event_time="19:00",
        updated_keys=set(),  # anchors existed before; restatements must be duplicates
    )
    assert "Дата: 2026-02-12" not in new_added
    assert "Время: 19:00" not in new_added
    assert any("12 февраля" in s for s in new_dup)
    assert any("19:00" in s.lower() for s in new_dup)
