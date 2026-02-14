import pytest

from db import Database
from models import Event
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


def _make_event(event_id: int, **overrides: object) -> Event:
    payload = {
        "id": event_id,
        "title": "Лорд Фаунтлерой",
        "description": "Спектакль в драмтеатре.",
        "date": "2026-02-05",
        "time": "19:00",
        "location_name": "Драматический театр",
        "city": "Калининград",
        "source_text": "старый текст",
    }
    payload.update(overrides)
    return Event(**payload)


@pytest.mark.asyncio
async def test_smart_update_falls_back_to_4o_when_gemma_unavailable(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        await session.commit()

    # Simulate Gemma being unavailable; Smart Update must fall back to 4o.
    monkeypatch.setenv("SMART_UPDATE_GEMMA_RETRIES", "1")
    monkeypatch.setenv("SMART_UPDATE_GEMMA_RETRY_BASE_SEC", "0.01")
    monkeypatch.setattr(su, "_get_gemma_client", lambda: None)

    async def _fake_ask_4o(text: str, **kwargs):  # noqa: ANN001 - test helper
        # The merge prompt asks for JSON with at least description/added_facts/duplicate_facts/skipped_conflicts.
        return (
            "{"
            "\"title\": null,"
            "\"description\": \"Спектакль «Лорд Фаунтлерой». Прекрасный дуэт Александра Егорова и Павла Самоловова.\","
            "\"ticket_link\": null,"
            "\"ticket_price_min\": null,"
            "\"ticket_price_max\": null,"
            "\"ticket_status\": null,"
            "\"added_facts\": [\"Прекрасный дуэт Александра Егорова и Павла Самоловова.\"],"
            "\"duplicate_facts\": [],"
            "\"skipped_conflicts\": []"
            "}"
        )

    import main as main_mod

    monkeypatch.setattr(main_mod, "ask_4o", _fake_ask_4o)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://fallback/1",
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

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert "дуэт" in (ev.description or "").lower()


@pytest.mark.asyncio
async def test_smart_update_does_not_inject_sentences_when_llm_merge_returns_stale_text(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1, description="Спектакль в драмтеатре. Премия Арлекин-2010."))
        await session.commit()

    async def _stale_merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": None,
            "description": "Спектакль в драмтеатре. Премия Арлекин-2010.",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "skipped_conflicts": [],
        }

    monkeypatch.setattr(su, "_llm_merge_event", _stale_merge)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://stale-merge/1",
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

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "merged"

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        # Requirement: no deterministic injection of raw sentences. Only LLM output is used.
        assert "дуэт" not in (ev.description or "").lower()


@pytest.mark.asyncio
async def test_smart_update_does_not_fallback_to_verbatim_source_text_when_merge_too_short(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    long_source = ("Подробное описание спектакля. " * 80).strip()
    async with db.get_session() as session:
        session.add(
            _make_event(
                1,
                title="Мёртвые души",
                description="Коротко.",
                source_text=long_source,
            )
        )
        await session.commit()

    async def _short_merge(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": None,
            "description": "Очень кратко.",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_full_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _short_merge)
    monkeypatch.setattr(su, "_rewrite_description_full_from_sources", _no_full_rewrite)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="test://short-merge/1",
        source_text="07.02 | Мёртвые души",
        raw_excerpt="",
        title="Мёртвые души",
        date="2026-02-05",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
        # Force merge even though candidate text is short.
        ticket_link="https://example.com/tickets",
    )

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "merged"

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert (ev.description or "").strip() == "Очень кратко."

@pytest.mark.asyncio
async def test_create_prefers_full_source_text_when_rewrite_unavailable_for_long_post(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _no_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)

    source_text = (
        "Благодарим каждого, кто готов стать частью события — вместе мы можем сделать мир немного лучше. "
        "Собираем книги для пациентов регионального реабилитационного центра «Новые горизонты». "
        "Книги помогают находить смысл, поддерживают и вдохновляют в период восстановления. "
        "Мы принимаем художественную литературу, познавательные книги и журналы, которые могут стать источником радости. "
        "Принести книги можно до 13 февраля по адресу: Калининград, проспект Мира 9/11, отдел библиотечного обслуживания. "
        "Работаем ежедневно с 10:00 до 18:00 (кроме пятницы). "
        "Давайте подарим надежду и поддержку тем, кто в этом особенно нуждается."
    )
    raw_excerpt = (
        "Принести книги можно до 13 февраля по адресу: Калининград, проспект Мира 9/11, "
        "отдел библиотечного обслуживания."
    )

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/kaliningradlibrary/2037",
        source_text=source_text,
        raw_excerpt=raw_excerpt,
        title="Книгодарение",
        date="2026-02-13",
        time="10:00-18:00",
        location_name="Научная библиотека",
        city="Калининград",
    )

    res = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        ev = await session.get(Event, res.event_id)
        assert ev is not None
        assert "реабилитационного центра" in (ev.description or "").lower()
        assert len(ev.description or "") > len(raw_excerpt)


@pytest.mark.asyncio
async def test_rewrite_short_telegram_source_is_not_overexpanded(monkeypatch):
    source_text = (
        "12.02 | Фигаро\n"
        "Нескучная французская классика. Настоящий театральный хит, проверенный временем."
    )
    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3782",
        source_text=source_text,
        raw_excerpt=source_text,
        title="Фигаро",
        date="2026-02-12",
        time="",
        location_name="Драматический театр",
        city="Калининград",
    )

    async def _long_rewrite(*args, **kwargs):  # noqa: ANN001 - test helper
        return (
            "В Калининградском драматическом театре состоится показ спектакля «Фигаро». "
            "Это постановка, которая сочетает классику и современный сценический язык.\n\n"
            "Зрителей ждёт насыщенная актёрская игра, работа с деталями, выразительная сценография "
            "и внимание к драматургии. "
        ) * 8

    monkeypatch.setattr(su, "_ask_gemma_text", _long_rewrite)
    monkeypatch.setattr(su, "SMART_UPDATE_LLM", "gemma")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    rewritten = await su._rewrite_description_journalistic(candidate)
    assert rewritten is not None

    cap = max(260, int(len(source_text.strip()) * 1.9) + 120)
    assert len(rewritten) <= cap
    # We don't require preserving exact source phrasing: rewrite must be non-verbatim.
    # But it should still clearly mention the event and avoid schedule-line noise.
    assert "Фигаро" in rewritten
    assert "12.02" not in rewritten


@pytest.mark.asyncio
async def test_merge_rejects_unrelated_llm_title_from_noncanonical_source(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            _make_event(
                1,
                title="Фигаро",
                description="Спектакль «Фигаро».",
                date="2026-02-12",
                time="19:00",
                location_name="Драматический театр",
            )
        )
        await session.commit()

    async def _merge_bad_title(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": 'Концерт "Музыка Времён" в Большом зале филармонии',
            "description": "Спектакль «Фигаро».",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_merge_event", _merge_bad_title)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/dramteatr39/3816",
        source_text="12.02 | Фигаро",
        raw_excerpt="12.02 | Фигаро",
        title="Фигаро",
        date="2026-02-12",
        time="",
        location_name="Драматический театр",
        city="Калининград",
        ticket_link="https://example.com/figaro-ticket",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "merged"
    assert any("Заголовок отклонён" in s for s in (result.skipped_conflicts or []))

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert ev.title == "Фигаро"


@pytest.mark.asyncio
async def test_merge_allows_parser_to_restore_title_after_pollution(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(
            _make_event(
                1,
                title='Концерт "Музыка Времён"',
                description="Повреждённое описание.",
                date="2026-02-12",
                time="19:00",
                location_name="Драматический театр",
            )
        )
        await session.commit()

    async def _match_event(*args, **kwargs):  # noqa: ANN001 - test helper
        return 1, 0.99, "test_match"

    async def _merge_fix_title(*args, **kwargs):  # noqa: ANN001 - test helper
        return {
            "title": "Фигаро",
            "description": "Спектакль «Фигаро».",
            "ticket_link": None,
            "ticket_price_min": None,
            "ticket_price_max": None,
            "ticket_status": None,
            "added_facts": [],
            "duplicate_facts": [],
            "conflict_facts": [],
            "skipped_conflicts": [],
        }

    async def _no_topics(*args, **kwargs):  # noqa: ANN001 - test helper
        return None

    monkeypatch.setattr(su, "_llm_match_event", _match_event)
    monkeypatch.setattr(su, "_llm_merge_event", _merge_fix_title)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)

    candidate = EventCandidate(
        source_type="parser:dramteatr",
        source_url="https://dramteatr39.ru/spektakli/figaro",
        source_text="О спектакле «Фигаро»",
        raw_excerpt="О спектакле «Фигаро»",
        title="Фигаро",
        date="2026-02-12",
        time="19:00",
        location_name="Драматический театр",
        city="Калининград",
        ticket_link="https://dramteatr39.ru/spektakli/figaro",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "merged"

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert ev.title == "Фигаро"
