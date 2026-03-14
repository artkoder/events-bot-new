from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import Event, EventSource
import smart_event_update as su
from smart_event_update import EventCandidate, smart_event_update


async def _no_merge(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return None


async def _no_rewrite(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return None


async def _no_digest(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return None


async def _no_short(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return None


async def _no_topics(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return None


async def _no_holidays(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return False


async def _minimal_create_bundle(*_args, **_kwargs):  # noqa: ANN001 - test helper
    return {
        "title": None,
        "description": "Тестовое описание.",
        "facts": [],
        "search_digest": None,
        "short_description": None,
    }


def _base_event(**overrides: object) -> Event:
    payload = {
        "title": "TEST",
        "description": "Описание.",
        "date": "2026-03-07",
        "time": "19:00",
        "location_name": "Тестовая площадка",
        "city": "Калининград",
        "source_text": "Тестовый анонс.",
    }
    payload.update(overrides)
    return Event(**payload)


async def _patch_llm_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", True)
    monkeypatch.setattr(su, "_get_gemma_client", lambda: None)
    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_build_short_description", _no_short)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _no_holidays)


async def _patch_llm_match_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)
    monkeypatch.setattr(su, "_get_gemma_client", lambda: None)
    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_build_short_description", _no_short)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _no_holidays)
    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _minimal_create_bundle)

    async def _must_not_match(*_args, **_kwargs):  # noqa: ANN001 - test helper
        raise AssertionError("LLM shortlist match must not run for blocked pair")

    monkeypatch.setattr(su, "_llm_match_or_create_bundle", _must_not_match)


async def _patch_llm_bundle(monkeypatch: pytest.MonkeyPatch, combo_func) -> None:  # noqa: ANN001
    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)
    monkeypatch.setattr(su, "_get_gemma_client", lambda: None)
    monkeypatch.setattr(su, "_llm_merge_event", _no_merge)
    monkeypatch.setattr(su, "_rewrite_description_journalistic", _no_rewrite)
    monkeypatch.setattr(su, "_llm_build_search_digest", _no_digest)
    monkeypatch.setattr(su, "_llm_build_short_description", _no_short)
    monkeypatch.setattr(su, "_classify_topics", _no_topics)
    monkeypatch.setattr(su, "_apply_holiday_festival_mapping", _no_holidays)
    monkeypatch.setattr(su, "_llm_create_description_facts_and_digest", _minimal_create_bundle)
    monkeypatch.setattr(su, "_llm_match_or_create_bundle", combo_func)


@pytest.mark.asyncio
async def test_same_post_longrun_exact_title_merges_same_source_time_noise(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    source_url = "https://vk.com/wall-1_100"
    source_text = "Выставка проходит с экскурсией в 12:00 и 15:00. До 30 апреля."
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Женственность через века",
                date="2026-03-07",
                time="12:00",
                end_date="2026-04-30",
                location_name="Музей TEST",
                source_text=source_text,
                source_post_url=source_url,
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="vk",
                source_url=source_url,
                source_text=source_text,
                trust_level="medium",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url=source_url,
        source_text=source_text,
        raw_excerpt="Выставка с экскурсиями.",
        title="Женственность через века",
        date="2026-03-07",
        time="15:00",
        end_date="2026-04-30",
        location_name="Музей TEST",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_same_post_longrun_exact_title_survives_time_filtered_shortlist(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    source_url = "https://vk.com/wall-152679358_23836"
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Женственность через века",
                date="2026-03-05",
                time="12:00",
                end_date="2026-06-05",
                location_name="Информационно-туристический центр",
                city="Черняховск",
                event_type="выставка",
                source_text="Выставка работает до 5 июня. Экскурсии в 12:00 и 15:00.",
                source_post_url=source_url,
                source_vk_post_url=source_url,
            )
        )
        session.add(
            _base_event(
                id=2,
                title="Другая выставка",
                date="2026-03-05",
                time="15:00",
                location_name="Информационно-туристический центр",
                city="Черняховск",
                event_type="выставка",
                source_text="Отвлекающая выставка на тот же слот.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url=source_url,
        source_text="Выставка работает до 5 июня. Экскурсии в 12:00 и 15:00.",
        raw_excerpt="Экскурсия по выставке.",
        title="Женственность через века",
        date="2026-03-05",
        time="15:00",
        end_date="2026-06-05",
        location_name="Информационно-туристический центр",
        city="Черняховск",
        event_type="выставка",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_cross_source_longrun_exact_title_merges_later_in_period_mention(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    museum = "Музей Изобразительных искусств, Ленинский проспект 83, Калининград"
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Путешествие Матрешки",
                date="2026-03-05",
                time="",
                end_date="2026-04-05",
                location_name=museum,
                location_address="Ленинский проспект 83",
                city="Калининград",
                event_type="выставка",
                source_text="Выставка «Путешествие Матрешки» работает по 5 апреля.",
                source_post_url="https://vk.com/wall-9118984_23492",
                source_vk_post_url="https://vk.com/wall-9118984_23492",
            )
        )
        session.add(
            _base_event(
                id=2,
                title="Другая выставка в музее",
                date="2026-03-10",
                time="",
                end_date="2026-04-10",
                location_name=museum,
                city="Калининград",
                event_type="выставка",
                source_text="Другая выставка в музее.",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/kaliningradartmuseum/7748",
        source_text=(
            "Благотворительная выставка «Путешествие Матрешки» продолжается. "
            "Красны девицы гостят в музее до 5 апреля."
        ),
        raw_excerpt="Выставка в музее до 5 апреля.",
        title="Путешествие Матрешки",
        date="2026-03-10",
        time="",
        end_date="2026-04-05",
        location_name="Музей",
        city="Калининград",
        event_type="выставка",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1

    async with db.get_session() as session:
        source_row = (
            await session.execute(
                select(EventSource).where(
                    EventSource.source_url == "https://t.me/kaliningradartmuseum/7748"
                )
            )
        ).scalar_one()
        assert source_row.event_id == 1


@pytest.mark.asyncio
async def test_same_source_anchor_message_id_merges_without_event_source_row(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    source_url = "https://vk.com/wall-212233232_1680"
    source_text = "14 марта в ОЦК ТеплоСеть пройдёт питчинг идей и клубов."
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Питчинг идей и клубов в «ТеплоСети»",
                date="2026-03-14",
                time="18:00",
                location_name="ОЦК ТеплоСеть",
                city="Советск",
                source_text=source_text,
                source_post_url=source_url,
                source_vk_post_url=source_url,
                source_chat_id=212233232,
                source_message_id=1680,
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url=source_url,
        source_text=source_text,
        raw_excerpt="Питчинг идей и клубов.",
        title="Питчинг идей и клубов в ТеплоСети",
        date="2026-03-14",
        time="18:00",
        location_name="ОЦК ТеплоСеть",
        city="Советск",
        source_chat_id=212233232,
        source_message_id=1680,
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_cross_source_exact_match_merges_emoji_prefixed_titles_without_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="✨ Пластический спектакль «Щелкунчик»",
                date="2026-12-14",
                time="15:00",
                location_name="Историко-художественный музей",
                source_text="Пластический спектакль «Щелкунчик».",
                source_post_url="https://vk.com/wall-1_200",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/kenigevents/2304",
        source_text="Пластический спектакль «Щелкунчик».",
        raw_excerpt="Спектакль в музее.",
        title="🎭 Пластический спектакль «Щелкунчик»",
        date="2026-12-14",
        time="15:00",
        location_name="Историко-художественный музей, Калининград",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_specific_ticket_same_slot_merges_only_via_narrow_rule(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    ticket = "https://signalcommunity.timepad.ru/event/3860844/"
    shared_text = "7 марта показ фильма «Маленькие женщины» в киноклубе."
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Сто семнадцатый показ киноклуба westside movieclub",
                date="2026-03-07",
                time="19:30",
                location_name="Сигнал",
                ticket_link=ticket,
                source_text=shared_text,
                source_post_url="https://t.me/signalkld/9892",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-218351015_153",
        source_text=shared_text,
        raw_excerpt="Показ фильма.",
        title="Маленькие женщины",
        date="2026-03-07",
        time="19:30",
        location_name="Сигнал",
        city="Калининград",
        ticket_link=ticket,
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_doors_start_ticket_bridge_merges_bar_sovetov_pair_without_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    shared_ticket = "https://t.me/stolik_na_standup_bot"
    shared_text = "Сбор гостей 19:30. Начало 20:00. Громкая связь в Bar Sovetov."
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Громкая связь: комедийное шоу",
                date="2026-03-06",
                time="20:00",
                location_name="Бар Sovetov, Мира 118, Калининград",
                ticket_link=shared_ticket,
                source_text=shared_text,
                source_post_url="https://vk.com/wall-214027639_10783",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/locostandup/3171",
        source_text=shared_text,
        raw_excerpt="Техническая вечеринка от LOCO Stand Up Club.",
        title="Громкая связь: техническая вечеринка от LOCO Stand Up Club",
        date="2026-03-06",
        time="19:30",
        location_name="Bar Sovetov",
        city="Калининград",
        ticket_link="http://t.me/stolik_na_standup_bot",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_generic_ticket_false_friend_blocks_match_path_before_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_match_must_not_run(monkeypatch)

    ticket = "https://tickets.sobor-kaliningrad.ru/scheme/541EA644B65930C2DFFDACBA44D4660A28137E01"
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="🎶 Английская придворная культура XVII века",
                date="2026-02-01",
                time="18:00",
                location_name="Кафедральный собор",
                ticket_link=ticket,
                source_text="Органный концерт.",
                source_post_url=ticket,
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_300",
        source_text="Сказочная музыка барокко.",
        raw_excerpt="Новый концерт.",
        title="🎭 Королева фей",
        date="2026-02-01",
        time="18:00",
        location_name="Кафедральный собор",
        city="Калининград",
        ticket_link=ticket,
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        events = (await session.execute(select(Event).order_by(Event.id))).scalars().all()
        assert len(events) == 2


@pytest.mark.asyncio
async def test_multi_event_source_blocker_blocks_same_source_program_false_merge_before_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_match_must_not_run(monkeypatch)

    source_url = "https://vk.com/wall-9118984_23596"
    ticket = "https://vk.cc/cV22qy"
    titles = [
        "8 Марта в Музее изобразительных искусств",
        "Акция «Вам, любимые!» в Музее изобразительных искусств",
        "Бесплатная экскурсия в Музей изобразительных искусств",
        "Весенний мастер-класс в Музее изобразительных искусств",
    ]
    async with db.get_session() as session:
        for idx, title in enumerate(titles, start=1):
            session.add(
                _base_event(
                    id=idx,
                    title=title,
                    date="2026-03-06",
                    time="",
                    location_name="Музей Изобразительных искусств, Ленинский проспект 83, Калининград",
                    ticket_link=ticket,
                    source_text="Праздничная программа музея.",
                    source_post_url=source_url,
                )
            )
            session.add(
                EventSource(
                    event_id=idx,
                    source_type="vk",
                    source_url=source_url,
                    source_text="Праздничная программа музея.",
                    trust_level="medium",
                )
            )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url=source_url,
        source_text="Праздничная программа музея.",
        raw_excerpt="Ещё один child event из той же программы.",
        title="Лекция о весеннем искусстве",
        date="2026-03-06",
        time="",
        location_name="Музей Изобразительных искусств, Ленинский проспект 83, Калининград",
        city="Калининград",
        ticket_link=ticket,
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None

    async with db.get_session() as session:
        events = (await session.execute(select(Event).order_by(Event.id))).scalars().all()
        assert len(events) == 5


@pytest.mark.asyncio
async def test_cross_sibling_guard_redirects_wrong_museum_sibling_before_merge(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def _wrong_combo(*_args, **_kwargs):  # noqa: ANN001 - test helper
        return {
            "action": "match",
            "match_event_id": 1,
            "confidence": 0.97,
            "reason_short": "wrong_sibling_for_test",
        }

    await _patch_llm_bundle(monkeypatch, _wrong_combo)
    monkeypatch.setattr(su, "_deterministic_exact_title_match", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(su, "_deterministic_related_title_anchor_match", lambda *_args, **_kwargs: None)

    museum = "Музей Изобразительных искусств, Ленинский проспект 83, Калининград"
    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Акция «Музей. Музы и творцы» в Музее изобразительных искусств",
                date="2026-03-07",
                time="",
                end_date="2026-03-09",
                location_name=museum,
                source_text="Праздничная программа музея.",
                source_post_url="https://vk.com/wall-9118984_23596",
            )
        )
        session.add(
            EventSource(
                event_id=1,
                source_type="vk",
                source_url="https://vk.com/wall-9118984_23596",
                source_text="Праздничная программа музея.",
                trust_level="medium",
            )
        )
        session.add(
            _base_event(
                id=2,
                title="Путешествие матрешки",
                date="2026-03-07",
                time="",
                end_date="2026-04-05",
                location_name=museum,
                source_text="Выставка «Путешествие Матрешки» работает по 5 апреля.",
                source_post_url="https://vk.com/wall-9118984_23492",
            )
        )
        session.add(
            EventSource(
                event_id=2,
                source_type="vk",
                source_url="https://vk.com/wall-9118984_23492",
                source_text="Выставка «Путешествие Матрешки» работает по 5 апреля.",
                trust_level="medium",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-9118984_23524",
        source_text=(
            "Сюжет о выставке «Путешествие Матрешки», выставка открыта по 5 апреля. "
            "Вход свободный при покупке билета в музей."
        ),
        raw_excerpt="Интерактивная экспозиция в музее.",
        title="Путешествие Матрешки: Интерактивная экспозиция",
        date="2026-03-07",
        time="",
        end_date="2026-04-05",
        location_name=museum,
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 2

    async with db.get_session() as session:
        source_row = (
            await session.execute(
                select(EventSource).where(EventSource.source_url == "https://vk.com/wall-9118984_23524")
            )
        ).scalar_one()
        assert source_row.event_id == 2


@pytest.mark.asyncio
async def test_cross_source_exact_match_merges_vistynets_pair_with_city_noise(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Ярмарка «Вкусов Виштынецкой возвышенности»",
                date="2026-03-07",
                time="12:00",
                end_date="2026-04-07",
                location_name="Дизайн-резиденция Gumbinnen, Ленина 29, Гусев",
                city="Гусев",
                source_text="Ярмарка локальных продуктов в Gumbinnen.",
                source_post_url="https://vk.com/wall-211015009_857",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/gumbinnen/1181",
        source_text="Ярмарка локальных продуктов проекта «Вкусы Виштынецкой возвышенности».",
        raw_excerpt="Мини-ярмарка в Gumbinnen.",
        title="Ярмарка «Вкусов Виштынецкой возвышенности»",
        date="2026-03-07",
        time="12:00",
        end_date="2026-04-07",
        location_name="Дизайн-резиденция Gumbinnen",
        city="Калининград",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "merged"
    assert res.event_id == 1


@pytest.mark.asyncio
async def test_copy_post_ticket_same_day_merges_multi_event_reposts_with_city_noise(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    first_source_url = "https://vk.com/wall-30777579_14694"
    first_source_text = (
        "В марте в Калининградскую область приедут федеральные специалисты по детской онкологии.\n"
        "Регистрация на прием — clck.ru/3RhSn2\n"
        "23 марта — Светлогорск\n"
        "24 марта — Калининград\n"
        "26 марта — Зеленоградск\n"
    )
    second_source_url = "https://vk.com/wall-211997788_2805"
    second_source_text = (
        "Бесплатные приёмы детских онкологов в Калининградской области.\n"
        "В марте в Калининградскую область приедут федеральные специалисты по детской онкологии.\n"
        "Регистрация на прием — https://clck.ru/3RhSn2\n"
        "23 марта — Светлогорск\n"
        "24 марта — Калининград\n"
        "26 марта — Зеленоградск\n"
    )

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Бесплатные консультации детских онкологов",
                date="2026-03-23",
                time="",
                city="Светлогорск",
                location_name="Светлогорск",
                ticket_link="clck.ru/3RhSn2",
                source_text=first_source_text,
                source_post_url=first_source_url,
                source_vk_post_url=first_source_url,
            )
        )
        session.add(
            _base_event(
                id=2,
                title="🩺 Бесплатные приемы детских онкологов",
                date="2026-03-24",
                time="",
                city="Калининград",
                location_name="Калининград Сити Джаз Клуб, Мира 33-35, Калининград",
                location_address="Мира 33-35",
                ticket_link="https://clck.ru/3RhSn2",
                source_text=first_source_text,
                source_post_url=first_source_url,
                source_vk_post_url=first_source_url,
            )
        )
        session.add(
            _base_event(
                id=3,
                title="Бесплатные консультации детских онкологов в Зеленоградске",
                date="2026-03-26",
                time="",
                city="Зеленоградск",
                location_name="Зеленоградск",
                ticket_link="clck.ru/3RhSn2",
                source_text=first_source_text,
                source_post_url=first_source_url,
                source_vk_post_url=first_source_url,
            )
        )
        session.add_all(
            [
                EventSource(
                    event_id=1,
                    source_type="vk",
                    source_url=first_source_url,
                    source_text=first_source_text,
                    trust_level="medium",
                ),
                EventSource(
                    event_id=2,
                    source_type="vk",
                    source_url=first_source_url,
                    source_text=first_source_text,
                    trust_level="medium",
                ),
                EventSource(
                    event_id=3,
                    source_type="vk",
                    source_url=first_source_url,
                    source_text=first_source_text,
                    trust_level="medium",
                ),
            ]
        )
        await session.commit()

    cases = [
        (
            "Бесплатные консультации детских онкологов",
            "2026-03-23",
            1,
        ),
        (
            "Бесплатные консультации детских онкологов",
            "2026-03-24",
            2,
        ),
        (
            "Бесплатные консультации детских онкологов",
            "2026-03-26",
            3,
        ),
    ]

    for title, event_date, expected_event_id in cases:
        candidate = EventCandidate(
            source_type="vk",
            source_url=second_source_url,
            source_text=second_source_text,
            raw_excerpt="Бесплатные приёмы детских онкологов в Калининградской области.",
            title=title,
            date=event_date,
            time="",
            city="Калининград",
            location_name="Сигнал",
            location_address="Леонова 22",
            ticket_link="https://clck.ru/3RhSn2",
            trust_level="medium",
        )

        res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
        assert res.status == "merged"
        assert res.event_id == expected_event_id

    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(EventSource.event_id).where(EventSource.source_url == second_source_url)
            )
        ).scalars().all()
        assert set(rows) == {1, 2, 3}


@pytest.mark.asyncio
async def test_zoo_generic_excursion_false_friend_does_not_merge_without_llm(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    await _patch_llm_off(monkeypatch)

    async with db.get_session() as session:
        session.add(
            _base_event(
                id=1,
                title="Экскурсия «Тайны панциря и чешуи, или О тех, кого не любят»",
                date="2026-03-07",
                time="11:00",
                location_name="Калининградский зоопарк, пр-т Мира 26, Калининград",
                city="Калининград",
                ticket_link="https://vk.cc/cUYxJb",
                source_text="Конкретная экскурсия про рептилий из цикла «Другой зоопарк».",
                source_post_url="https://vk.com/wall-48383763_39377",
            )
        )
        await session.commit()

    candidate = EventCandidate(
        source_type="telegram",
        source_url="https://t.me/example/1",
        source_text="Посетители смогут самостоятельно изучить территорию по специальной карте.",
        raw_excerpt="Самостоятельное посещение зоопарка.",
        title="Экскурсии в зоопарке Калининграда",
        date="2026-03-07",
        time="11:00",
        location_name="Зоопарк",
        city="Калининград",
        ticket_link="https://vk.cc/cVbnez",
        trust_level="medium",
    )

    res = await smart_event_update(db, candidate, check_source_url=False, schedule_tasks=False)
    assert res.status == "created"
    assert res.event_id is not None and res.event_id != 1
