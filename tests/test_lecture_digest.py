import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import main
from main import Database, Event
import logging
import digests
from digests import (
    build_lectures_digest_candidates,
    build_masterclasses_digest_candidates,
    build_masterclasses_digest_preview,
    build_exhibitions_digest_candidates,
    build_exhibitions_digest_preview,
    build_psychology_digest_candidates,
    build_psychology_digest_preview,
    build_science_pop_digest_candidates,
    build_science_pop_digest_preview,
    build_networking_digest_candidates,
    build_entertainment_digest_candidates,
    build_markets_digest_candidates,
    build_theatre_classic_digest_candidates,
    build_theatre_modern_digest_candidates,
    build_meetups_digest_candidates,
    build_meetups_digest_preview,
    build_movies_digest_candidates,
    compose_digest_intro_via_4o,
    compose_masterclasses_intro_via_4o,
    compose_exhibitions_intro_via_4o,
    compose_meetups_intro_via_4o,
    aggregate_digest_topics,
    normalize_topics,
    format_event_line_html,
    pick_display_link,
    normalize_titles_via_4o,
    assemble_compact_caption,
    visible_caption_len,
    compose_digest_caption,
    attach_caption_if_fits,
    _build_digest_preview,
    _normalize_title_fallback,
)
from aiogram import types


@pytest.mark.asyncio
async def test_compose_meetups_intro_prompt_mentions_anti_cliche(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_ask(prompt: str, *, max_tokens: int) -> str:
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    meetup = {
        "title": "Design Meetup",
        "description": "Discuss product design",
        "event_type": "meetup",
        "formats": ["лекция"],
    }

    await compose_meetups_intro_via_4o(1, 7, [meetup])

    prompt = captured["prompt"]
    for wording in digests.MEETUPS_INTRO_FORBIDDEN_WORDINGS:
        assert wording in prompt

    assert "Первая фраза должна начинаться с интригующего хука" in prompt


@pytest.mark.asyncio
async def test_build_lectures_digest_candidates_expand_to_14(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, time: str, title: str):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=time,
                location_name="x",
                source_text="s",
                event_type="лекция",
                source_post_url="http://example.com/" + title,
            )
            session.add(ev)

        # Event starting in less than 2h -> should be ignored
        add(0, "13:00", "early")
        # Five events within 7 days
        add(0, "15:00", "d0")
        add(1, "12:00", "d1")
        add(2, "12:00", "d2")
        add(3, "12:00", "d3")
        add(4, "12:00", "d4")
        # Two more beyond 7 days
        add(8, "12:00", "d8")
        add(9, "12:00", "d9")
        await session.commit()

    events, horizon = await build_lectures_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["d0", "d1", "d2", "d3", "d4", "d8", "d9"]


@pytest.mark.asyncio
async def test_build_masterclasses_digest_candidates(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, time: str, title: str):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=time,
                location_name="x",
                source_text="s",
                event_type="мастер-класс",
                source_post_url="http://example.com/" + title,
            )
            session.add(ev)

        add(0, "15:00", "m0")
        add(3, "12:00", "m3")
        add(8, "12:00", "m8")
        await session.commit()

    events, horizon = await build_masterclasses_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["m0", "m3", "m8"]


@pytest.mark.asyncio
async def test_build_psychology_digest_candidates_filters_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(title: str, *, topics: list[str], offset_days: int = 1):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time="18:00",
                location_name="x",
                source_text="s",
                source_post_url=f"http://example.com/{title}",
                event_type="лекция",
                topics=topics,
            )
            session.add(ev)

        add("Psych 1", topics=["Психология"])
        add("History", topics=["История"])
        add("Psych 2", topics=["mental health"], offset_days=2)
        await session.commit()

    events, horizon = await build_psychology_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["Psych 1", "Psych 2"]


@pytest.mark.asyncio
async def test_build_science_pop_digest_candidates_filters_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(title: str, *, topics: list[str], offset_days: int = 1):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time="19:00",
                location_name="x",
                source_text="s",
                source_post_url=f"http://example.com/{title}",
                event_type="лекция",
                topics=topics,
            )
            session.add(ev)

        add("Science Pop", topics=["Научпоп"])
        add("Tech Talk", topics=["технологии"], offset_days=2)
        add("Other", topics=["лекция"], offset_days=3)
        await session.commit()

    events, horizon = await build_science_pop_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["Science Pop", "Tech Talk"]


@pytest.mark.asyncio
async def test_build_networking_digest_candidates_filters_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 10, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Business Breakfast",
                description="",
                date="2025-05-02",
                time="10:00",
                location_name="Cafe",
                source_text="s",
                source_post_url="http://example.com/breakfast",
                topics=["business breakfast"],
            )
        )
        session.add(
            Event(
                title="Other Meetup",
                description="",
                date="2025-05-03",
                time="12:00",
                location_name="Hall",
                source_text="s",
                source_post_url="http://example.com/other",
                topics=["LECTURES"],
            )
        )
        await session.commit()

    events, horizon = await build_networking_digest_candidates(db, now)
    assert horizon == 14
    assert [e.title for e in events] == ["Business Breakfast"]


@pytest.mark.asyncio
async def test_build_entertainment_digest_candidates_merges_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        def add(title: str, day_offset: int, time: str, topics: list[str]):
            dt = now + timedelta(days=day_offset)
            session.add(
                Event(
                    title=title,
                    description="",
                    date=dt.strftime("%Y-%m-%d"),
                    time=time,
                    location_name="Club",
                    source_text="s",
                    source_post_url=f"http://example.com/{title}",
                    topics=topics,
                )
            )

        add("Standup Night", 1, "19:00", ["STANDUP"])
        add("Quiz Battle", 2, "20:00", ["QUIZ_GAMES"])
        add("Party Mix", 3, "18:00", ["STANDUP", "QUIZ_GAMES"])
        await session.commit()

    events, horizon = await build_entertainment_digest_candidates(db, now)
    titles = [e.title for e in events]
    assert horizon == 14
    assert titles == ["Standup Night", "Quiz Battle", "Party Mix"]


@pytest.mark.asyncio
async def test_build_markets_digest_candidates_filters_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Spring Market",
                description="",
                date="2025-05-02",
                time="12:00",
                location_name="Loft",
                source_text="s",
                source_post_url="http://example.com/market",
                topics=["HANDMADE"],
            )
        )
        session.add(
            Event(
                title="Concert",
                description="",
                date="2025-05-02",
                time="15:00",
                location_name="Stage",
                source_text="s",
                source_post_url="http://example.com/concert",
                topics=["CONCERTS"],
            )
        )
        await session.commit()

    events, _ = await build_markets_digest_candidates(db, now)
    assert [e.title for e in events] == ["Spring Market"]


@pytest.mark.asyncio
async def test_build_theatre_classic_digest_candidates_filters_event_type(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Classic Drama",
                description="",
                date="2025-05-02",
                time="19:00",
                location_name="Theatre",
                source_text="s",
                source_post_url="http://example.com/drama",
                event_type="спектакль",
                topics=["THEATRE_CLASSIC"],
            )
        )
        session.add(
            Event(
                title="Classic Talk",
                description="",
                date="2025-05-03",
                time="19:00",
                location_name="Hall",
                source_text="s",
                source_post_url="http://example.com/talk",
                event_type="лекция",
                topics=["THEATRE_CLASSIC"],
            )
        )
        await session.commit()

    events, _ = await build_theatre_classic_digest_candidates(db, now)
    assert [e.title for e in events] == ["Classic Drama"]


@pytest.mark.asyncio
async def test_build_theatre_modern_digest_candidates_filters_event_type(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Modern Show",
                description="",
                date="2025-05-02",
                time="21:00",
                location_name="Stage",
                source_text="s",
                source_post_url="http://example.com/modern",
                event_type="спектакль",
                topics=["THEATRE_MODERN"],
            )
        )
        session.add(
            Event(
                title="Modern Lecture",
                description="",
                date="2025-05-03",
                time="19:00",
                location_name="Hall",
                source_text="s",
                source_post_url="http://example.com/modern-lecture",
                event_type="лекция",
                topics=["THEATRE_MODERN"],
            )
        )
        await session.commit()

    events, _ = await build_theatre_modern_digest_candidates(db, now)
    assert [e.title for e in events] == ["Modern Show"]


@pytest.mark.asyncio
async def test_build_meetups_digest_candidates_filters_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Book Club",
                description="",
                date="2025-05-02",
                time="18:00",
                location_name="Library",
                source_text="s",
                source_post_url="http://example.com/bookclub",
                topics=["book club"],
            )
        )
        session.add(
            Event(
                title="Other Event",
                description="",
                date="2025-05-02",
                time="20:00",
                location_name="Cafe",
                source_text="s",
                source_post_url="http://example.com/other",
                topics=["CONCERTS"],
            )
        )
        await session.commit()

    events, _ = await build_meetups_digest_candidates(db, now)
    assert [e.title for e in events] == ["Book Club"]


@pytest.mark.asyncio
async def test_build_movies_digest_candidates_filters_event_type(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 9, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Cinema Night",
                description="",
                date="2025-05-02",
                time="21:00",
                location_name="Cinema",
                source_text="s",
                source_post_url="http://example.com/cinema",
                event_type="кинопоказ",
            )
        )
        session.add(
            Event(
                title="Lecture",
                description="",
                date="2025-05-02",
                time="18:00",
                location_name="Hall",
                source_text="s",
                source_post_url="http://example.com/lecture",
                event_type="лекция",
            )
        )
        await session.commit()

    events, _ = await build_movies_digest_candidates(db, now)
    assert [e.title for e in events] == ["Cinema Night"]


@pytest.mark.asyncio
async def test_build_psychology_digest_preview_filters_topics(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        session.add(
            Event(
                title="Mindfulness",
                description="Keep calm",
                date="2025-05-02",
                time="18:30",
                location_name="loc",
                source_text="s",
                source_post_url="http://example.com/mind",
                event_type="лекция",
                topics=["Психология", "wellbeing"],
            )
        )
        session.add(
            Event(
                title="Other",
                description="No",
                date="2025-05-03",
                time="19:00",
                location_name="loc",
                source_text="s",
                source_post_url="http://example.com/other",
                event_type="лекция",
                topics=["История"],
            )
        )
        await session.commit()

    async def fake_normalize(titles, event_kind=None, events=None):
        return [{"title_clean": title, "emoji": ""} for title in titles]

    recorded_payload: list[list[dict[str, object]]] = []

    async def fake_psych_intro(n, horizon_days, payload):
        recorded_payload.append(payload)
        return "Psych intro"

    monkeypatch.setattr(digests, "normalize_titles_via_4o", fake_normalize)
    monkeypatch.setattr(digests, "compose_psychology_intro_via_4o", fake_psych_intro)

    intro, lines, horizon, events, norm_titles = await build_psychology_digest_preview(
        "dg", db, now
    )

    assert intro == "Psych intro"
    assert horizon == 14
    assert [e.title for e in events] == ["Mindfulness"]
    assert norm_titles == ["Mindfulness"]
    assert len(lines) == 1
    assert recorded_payload[-1] == [
        {
            "title": "Mindfulness",
            "description": "Keep calm",
            "topics": ["PSYCHOLOGY", "wellbeing"],
        }
    ]


@pytest.mark.asyncio
async def test_build_exhibitions_digest_candidates(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, time: str, title: str, *, duration_days: int = 2):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=time,
                location_name="x",
                source_text="s",
                event_type="выставка",
                source_post_url="http://example.com/" + title,
                end_date=(dt + timedelta(days=duration_days)).strftime("%Y-%m-%d"),
            )
            session.add(ev)

        add(-3, "10:00", "past", duration_days=7)
        add(0, "15:00", "e0")
        add(5, "12:00", "e5")
        add(9, "12:00", "e9")
        await session.commit()

    events, horizon = await build_exhibitions_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["e0", "past", "e5", "e9"]


@pytest.mark.asyncio
async def test_build_exhibitions_digest_candidates_includes_ongoing(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add_event(date_offset: int, *, end_offset: int, title: str, time: str = "12:00"):
            start = now + timedelta(days=date_offset)
            ev = Event(
                title=title,
                description="d",
                date=start.strftime("%Y-%m-%d"),
                time=time,
                location_name="x",
                source_text="s",
                event_type="выставка",
                source_post_url="http://example.com/" + title,
                end_date=(start + timedelta(days=end_offset)).strftime("%Y-%m-%d"),
            )
            session.add(ev)

        add_event(-5, end_offset=10, title="ongoing")
        add_event(0, end_offset=1, title="soon", time="15:00")
        add_event(0, end_offset=1, title="too-soon", time="13:00")
        await session.commit()

    events, horizon = await build_exhibitions_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert "ongoing" in titles
    assert "soon" in titles
    assert "too-soon" not in titles


@pytest.mark.asyncio
async def test_normalize_titles_exhibition_keeps_original():
    titles = [
        "🎨 Выставка «Солнечный луч»",
        "Выставка в Доме искусств",
    ]

    normalized = await normalize_titles_via_4o(titles, event_kind="exhibition")

    assert [item["emoji"] for item in normalized] == ["🎨", ""]
    cleaned = [item["title_clean"] for item in normalized]
    assert cleaned == ["Выставка «Солнечный луч»", "Выставка в Доме искусств"]
    assert all(not title.startswith("Лекция") for title in cleaned)


def test_normalize_title_fallback_keeps_non_name_prefix():
    title = "Исторический интенсив: история XX века"

    normalized = _normalize_title_fallback(title)

    assert normalized == {"emoji": "", "title_clean": title}


@pytest.mark.asyncio
async def test_build_digest_preview_exhibition_uses_clean_titles(monkeypatch):
    events = [
        SimpleNamespace(
            id=1,
            title="🎨 Выставка «Солнечный луч»",
            description="desc",
            date="2025-05-05",
            time="12:00",
            event_type="выставка",
            end_date="2025-05-20",
            source_post_url="https://example.com/post",
            telegraph_url=None,
            telegraph_path=None,
        )
    ]

    async def fake_candidates_builder(db, now, digest_id):
        return events, 14

    async def fake_intro(*args, **kwargs):
        return "intro"

    monkeypatch.setattr(digests, "compose_exhibitions_intro_via_4o", fake_intro)

    intro, lines, horizon, returned_events, norm_titles = await _build_digest_preview(
        "digest-id",
        db=None,
        now=datetime(2025, 5, 1, 12, 0),
        kind="exhibitions",
        event_noun="выставок",
        event_kind="exhibition",
        candidates_builder=fake_candidates_builder,
    )

    assert intro == "intro"
    assert horizon == 14
    assert returned_events == events
    assert norm_titles == ["Выставка «Солнечный луч»"]
    assert lines and "Выставка «Солнечный луч»" in lines[0]
    assert "Лекция" not in lines[0]


@pytest.mark.asyncio
async def test_build_science_pop_digest_preview_uses_generic_intro(monkeypatch):
    events = [
        SimpleNamespace(
            id=7,
            title="Научпоп: космос",
            description="Истории про космос",
            date="2025-05-05",
            time="18:00",
            event_type="лекция",
            source_post_url="https://example.com/science",
            telegraph_url=None,
            telegraph_path=None,
        )
    ]

    captured: dict[str, object] = {}

    async def fake_candidates(db, now, digest_id):
        return events, 7

    async def fake_intro(count, horizon, titles, *, event_noun):
        captured["count"] = count
        captured["horizon"] = horizon
        captured["titles"] = titles
        captured["event_noun"] = event_noun
        return "intro"

    async def fake_normalize(titles, *, event_kind, events):
        captured["event_kind"] = event_kind
        return [{"title_clean": t, "emoji": ""} for t in titles]

    monkeypatch.setattr(
        digests, "build_science_pop_digest_candidates", fake_candidates
    )
    monkeypatch.setattr(digests, "compose_digest_intro_via_4o", fake_intro)
    monkeypatch.setattr(digests, "normalize_titles_via_4o", fake_normalize)

    intro, lines, horizon, returned_events, norm_titles = (
        await build_science_pop_digest_preview(
            "digest-science",
            db=None,
            now=datetime(2025, 5, 1, 12, 0),
        )
    )

    assert intro == "intro"
    assert horizon == 7
    assert returned_events == events
    assert norm_titles == ["Научпоп: космос"]
    assert captured["event_noun"] == "научно-популярных событий"
    assert captured["event_kind"] == "science_pop"
    assert lines and "Научпоп: космос" in lines[0]


@pytest.mark.asyncio
async def test_build_meetups_digest_preview_adds_exhibition_context(monkeypatch):
    events = [
        SimpleNamespace(
            id=42,
            title="Творческая встреча «Диалог о живописи»",
            description="Встреча с художником и открытие выставки.",
            date="2025-05-05",
            time="18:00",
            event_type="выставка",
            source_post_url="https://example.com/meetup",
            telegraph_url=None,
            telegraph_path=None,
        )
    ]

    async def fake_candidates_builder(db, now, digest_id):
        return events, 14

    async def fake_intro(*args, **kwargs):
        return "intro"

    async def fake_ask(prompt, max_tokens=0):
        raise RuntimeError("no llm")

    monkeypatch.setattr("main.ask_4o", fake_ask)
    monkeypatch.setattr(digests, "compose_digest_intro_via_4o", fake_intro)
    monkeypatch.setattr(digests, "compose_meetups_intro_via_4o", fake_intro)

    intro, lines, horizon, returned_events, norm_titles = await _build_digest_preview(
        "digest-meetup",
        db=None,
        now=datetime(2025, 5, 1, 12, 0),
        kind="meetups",
        event_noun="встреч",
        event_kind="meetups",
        candidates_builder=fake_candidates_builder,
    )

    assert intro == "intro"
    assert horizon == 14
    assert returned_events == events
    assert norm_titles[0].endswith("— творческая встреча и открытие выставки")
    assert "— творческая встреча и открытие выставки" in lines[0]


@pytest.mark.asyncio
async def test_build_exhibitions_digest_candidates_keeps_past_start_in_horizon(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add_event(offset_days: int, *, duration_days: int, title: str):
            start = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=start.strftime("%Y-%m-%d"),
                time="15:00",
                location_name="x",
                source_text="s",
                event_type="выставка",
                source_post_url="http://example.com/" + title,
                end_date=(start + timedelta(days=duration_days)).strftime("%Y-%m-%d"),
            )
            session.add(ev)

        add_event(-2, duration_days=5, title="ongoing")
        for idx in range(8):
            duration = 0 if idx == 7 else 1
            add_event(idx, duration_days=duration, title=f"e{idx}")
        await session.commit()

    events, horizon = await build_exhibitions_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 7
    assert len(events) == 9
    assert "ongoing" in titles


@pytest.mark.asyncio
async def test_build_lectures_digest_candidates_limit(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        def add(offset_days: int, hour: int, title: str):
            dt = now + timedelta(days=offset_days)
            ev = Event(
                title=title,
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time=f"{hour:02d}:00",
                location_name="x",
                source_text="s",
                event_type="лекция",
                source_post_url="http://example.com/" + title,
            )
            session.add(ev)

        add(0, 13, "early")  # filtered by +2h rule
        idx = 0
        for day in range(7):
            for h in (15, 16):
                idx += 1
                add(day, h, f"e{idx}")
        await session.commit()

    events, horizon = await build_lectures_digest_candidates(db, now)

    assert horizon == 7
    assert len(events) == 9
    assert events[0].title == "e1"


def test_normalize_topics_distinguishes_theatre_subtypes():
    topics = [
        "театр",
        "классический спектакль",
        "Драма",
        "современный театр",
        "модерн",
        "experimental theatre",
    ]

    normalized = normalize_topics(topics)

    assert normalized == ["THEATRE", "THEATRE_CLASSIC", "THEATRE_MODERN"]


def test_normalize_topics_collapses_kaliningrad_synonyms():
    topics = [
        "Калининград",
        "урбанистика",
        "краеведческий",
        "#калининград",
    ]

    normalized = normalize_topics(topics)

    assert normalized == ["KRAEVEDENIE_KALININGRAD_OBLAST"]


@pytest.mark.asyncio
async def test_compose_intro_via_4o(monkeypatch, caplog):
    async def fake_ask(prompt, max_tokens=0):
        assert "дайджест" in prompt.lower()
        assert "лекций" in prompt
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)
    text = await compose_digest_intro_via_4o(2, 7, ["e1", "e2"])
    assert text == "интро"
    assert any("digest.intro.llm.request" in r.message for r in caplog.records)
    assert any("digest.intro.llm.response" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_compose_intro_via_4o_masterclass(monkeypatch):
    captured_prompt: dict[str, str] = {}

    async def fake_ask(prompt, max_tokens=0):
        captured_prompt["value"] = prompt
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    payload = [
        {"title": "Акварель", "description": "Рисуем пейзажи акварелью. 12+"},
        {"title": "Голос", "description": "Работа с голосом и дыханием."},
    ]

    text = await compose_masterclasses_intro_via_4o(2, 7, payload)
    assert text == "интро"

    prompt = captured_prompt["value"]
    assert "Рисуем пейзажи акварелью. 12+" in prompt
    assert "Работа с голосом" in prompt
    assert "реальные активности" in prompt
    assert "не выдумывай фактов" in prompt
    assert "эмодзи" in prompt
    assert "200 символов" in prompt
    assert "формат группы" in prompt
    assert "перечисли основные активности" in prompt.lower()
    assert "12+" in prompt


@pytest.mark.asyncio
async def test_compose_intro_via_4o_exhibition(monkeypatch):
    captured_prompt: dict[str, str] = {}

    async def fake_ask(prompt, max_tokens=0):
        captured_prompt["value"] = prompt
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    payload = [
        {
            "title": "Импрессионисты",
            "description": "Большая экспозиция французских художников.",
            "date_range": {"start": "2025-05-01", "end": "2025-05-20"},
        },
        {
            "title": "Графика",
            "description": "Современная графика из частных коллекций.",
            "date_range": {"start": "2025-05-05", "end": "2025-05-18"},
        },
    ]

    text = await compose_exhibitions_intro_via_4o(2, 7, payload)
    assert text == "интро"

    prompt = captured_prompt["value"]
    assert "Импрессионисты" in prompt
    assert "2025-05-01" in prompt and "2025-05-20" in prompt
    assert "Не пропустите" in prompt
    assert "заканчивается" in prompt
    assert "1–2 связных предложения" in prompt
    assert "без списков" in prompt
    assert "description" in prompt and "date_range" in prompt
    assert "даты окончания" in prompt
    assert "эмодзи" in prompt
    assert "не выдумывай детали" in prompt
    assert "Сохрани каркас" not in prompt
    assert "После тире подчеркни" not in prompt


    event = SimpleNamespace(
        id=1,
        title="🎨 Мастер-класс «Акварель»",
        description="Рисуем пейзажи акварелью. 12+",
        date="2025-05-02",
        time="15:00",
        location_name="Студия",
        source_post_url="https://example.com/post",
        telegraph_url=None,
        telegraph_path=None,
        event_type="мастер-класс",
    )

    captured_payload: dict[str, list] = {}

    async def fake_compose_masterclasses_intro(n, horizon_days, masterclasses):
        captured_payload["value"] = masterclasses
        return "готовое интро"

    async def fake_normalize(titles, *, event_kind="lecture", events=None):
        return [{"emoji": "🎨", "title_clean": "Акварель"} for _ in titles]

    async def fake_candidates(db, now, digest_id=None):
        return [event], 7

    monkeypatch.setattr(
        "digests.compose_masterclasses_intro_via_4o",
        fake_compose_masterclasses_intro,
    )
    monkeypatch.setattr("digests.normalize_titles_via_4o", fake_normalize)
    monkeypatch.setattr(
        "digests.build_masterclasses_digest_candidates", fake_candidates
    )
    monkeypatch.setattr("digests.pick_display_link", lambda ev: event.source_post_url)

    now = datetime(2025, 5, 1, 12, 0)
    intro, lines, horizon, events, norm_titles = await build_masterclasses_digest_preview(
        "digest-test", None, now
    )

    assert intro == "готовое интро"
    assert captured_payload["value"] == [
        {"title": "Акварель", "description": "Рисуем пейзажи акварелью. 12+"}
    ]
    assert horizon == 7
    assert norm_titles == ["Акварель"]
    assert len(lines) == 1
    assert events == [event]


@pytest.mark.asyncio
async def test_build_meetups_digest_preview_payload(monkeypatch):
    event = SimpleNamespace(
        id=5,
        title="🎬 Киноклуб «Весна»",
        description="Творческий вечер с режиссёром и обсуждением фильма.",
        date="2025-05-05",
        time="19:00",
        location_name="Дом культуры",
        source_post_url="https://example.com/meetup",
        telegraph_url=None,
        telegraph_path=None,
        event_type="клуб",
    )

    captured_payload: dict[str, object] = {}

    async def fake_compose_meetups_intro(n, horizon_days, meetups, tone_hint=None):
        captured_payload["value"] = meetups
        captured_payload["tone_hint"] = tone_hint
        return "интро про встречи"

    async def fake_candidates(db, now, digest_id=None):
        return [event], 7

    async def fake_normalize_titles(titles, event_kind="lecture", events=None):
        return [{"emoji": "🎬", "title_clean": "Киноклуб «Весна»"}]

    monkeypatch.setattr(
        "digests.compose_meetups_intro_via_4o", fake_compose_meetups_intro
    )
    monkeypatch.setattr(
        "digests.build_meetups_digest_candidates", fake_candidates
    )
    monkeypatch.setattr(
        "digests.normalize_titles_via_4o", fake_normalize_titles
    )
    monkeypatch.setattr("digests.pick_display_link", lambda ev: ev.source_post_url)

    now = datetime(2025, 5, 1, 12, 0)
    intro, lines, horizon, events, norm_titles = await build_meetups_digest_preview(
        "digest-meetups", None, now
    )

    assert intro == "интро про встречи"
    assert captured_payload["value"] == [
        {
            "title": "Киноклуб «Весна»",
            "description": "Творческий вечер с режиссёром и обсуждением фильма.",
            "event_type": "клуб",
            "formats": ["клуб", "творческий вечер"],
        }
    ]
    assert captured_payload["tone_hint"] == "простота+любопытство"
    assert horizon == 7
    assert norm_titles == ["Киноклуб «Весна»"]
    assert len(lines) == 1
    assert events == [event]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events_data, expected_hint",
    [
        (
            [
                {
                    "title": "Закулисье проекта",
                    "description": "Узнаете секреты впервые",
                    "event_type": "встреча",
                }
            ],
            "любопытство+интрига",
        ),
        (
            [
                {
                    "title": "Открытая встреча друзей",
                    "description": "",
                    "event_type": "встреча",
                },
                {
                    "title": "Тёплый круг",
                    "description": "Без подготовки узнаете редкие факты",
                    "event_type": "встреча",
                },
            ],
            "простота+любопытство",
        ),
        (
            [
                {
                    "title": "Секретная открытая встреча",
                    "description": "Без подготовки и немного секретов",
                    "event_type": "встреча",
                }
            ],
            "интрига+простота",
        ),
    ],
)
async def test_build_meetups_digest_preview_tone_hints(
    monkeypatch, events_data, expected_hint
):
    events = [
        SimpleNamespace(
            id=index + 1,
            title=data["title"],
            description=data["description"],
            date="2025-05-05",
            time="19:00",
            location_name="Локация",
            source_post_url=f"https://example.com/{index}",
            telegraph_url=None,
            telegraph_path=None,
            event_type=data["event_type"],
        )
        for index, data in enumerate(events_data)
    ]

    captured: dict[str, object] = {}

    async def fake_compose_meetups_intro(n, horizon_days, meetups, tone_hint=None):
        captured["tone_hint"] = tone_hint
        return "интро"

    async def fake_candidates(db, now, digest_id=None):
        return events, 7

    async def fake_normalize_titles(titles, event_kind="lecture", events=None):
        return [{"emoji": "", "title_clean": title} for title in titles]

    monkeypatch.setattr(
        "digests.compose_meetups_intro_via_4o", fake_compose_meetups_intro
    )
    monkeypatch.setattr(
        "digests.build_meetups_digest_candidates", fake_candidates
    )
    monkeypatch.setattr("digests.normalize_titles_via_4o", fake_normalize_titles)
    monkeypatch.setattr("digests.pick_display_link", lambda ev: ev.source_post_url)

    now = datetime(2025, 5, 1, 12, 0)
    intro, lines, horizon, returned_events, norm_titles = await build_meetups_digest_preview(
        "digest-meetups", None, now
    )

    assert intro == "интро"
    assert captured["tone_hint"] == expected_hint
    assert horizon == 7
    assert returned_events == events
    assert len(lines) == len(events)
    assert norm_titles == [item.title for item in events]


@pytest.mark.asyncio
async def test_compose_meetups_intro_without_clubs_emphasises_people(monkeypatch):
    captured_prompt: dict[str, str] = {}

    async def fake_ask(prompt, max_tokens=0):
        captured_prompt["value"] = prompt
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    payload = [
        {
            "title": "Встреча с автором",
            "description": "Обсуждаем новую книгу и задаём вопросы.",
            "event_type": "встреча",
            "formats": ["встреча"],
        }
    ]

    text = await compose_meetups_intro_via_4o(1, 7, payload, "простота+любопытство")
    assert text == "интро"

    prompt = captured_prompt["value"]
    assert "has_club=false" in prompt
    assert "живом общении" in prompt or "интересными людьми" in prompt
    assert "q&a" in prompt.lower()
    assert "Простота + любопытство" in prompt


@pytest.mark.asyncio
async def test_compose_meetups_intro_with_club_sets_flag(monkeypatch):
    captured_prompt: dict[str, str] = {}

    async def fake_ask(prompt, max_tokens=0):
        captured_prompt["value"] = prompt
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)

    payload = [
        {
            "title": "Киноклуб",
            "description": "Просмотр фильма",
            "event_type": "клуб",
            "formats": ["клуб"],
        }
    ]

    text = await compose_meetups_intro_via_4o(1, 14, payload, "любопытство+интрига")
    assert text == "интро"

    prompt = captured_prompt["value"]
    assert "has_club=true" in prompt
    assert "Любопытство + интрига" in prompt
    assert "q&a" in prompt.lower()
    assert "живое общение" in prompt or "интересными людьми" in prompt


@pytest.mark.asyncio
async def test_build_exhibitions_digest_preview(monkeypatch):
    event = SimpleNamespace(
        id=2,
        title="🖼️ Выставка «Импрессионисты»",
        description="Картины из музеев Европы.",
        date="2025-05-02",
        end_date="2025-05-20",
        time="10:00",
        location_name="Музей",
        source_post_url="https://example.com/exhibit",
        telegraph_url=None,
        telegraph_path=None,
        event_type="выставка",
    )

    captured_payload: dict[str, list] = {}

    async def fake_compose_exhibitions_intro(n, horizon_days, exhibitions):
        captured_payload["value"] = exhibitions
        return "интро про выставки"

    async def fake_candidates(db, now, digest_id=None):
        return [event], 14

    monkeypatch.setattr(
        "digests.compose_exhibitions_intro_via_4o",
        fake_compose_exhibitions_intro,
    )
    monkeypatch.setattr(
        "digests.build_exhibitions_digest_candidates", fake_candidates
    )
    monkeypatch.setattr("digests.pick_display_link", lambda ev: ev.source_post_url)

    now = datetime(2025, 5, 1, 12, 0)
    intro, lines, horizon, events, norm_titles = await build_exhibitions_digest_preview(
        "digest-exhibit", None, now
    )

    assert intro == "интро про выставки"
    assert captured_payload["value"] == [
        {
            "title": "Выставка «Импрессионисты»",
            "description": "Картины из музеев Европы.",
            "date_range": {"start": "2025-05-02", "end": "2025-05-20"},
        }
    ]
    assert horizon == 14
    assert norm_titles == ["Выставка «Импрессионисты»"]
    assert len(lines) == 1
    assert events == [event]


def test_visible_len_anchors():
    html = '<a href="https://very.long/url">Заголовок</a>'
    assert visible_caption_len(html) == len("Заголовок")
    html2 = '<a href="http://a">A</a> <b>B</b>'
    assert visible_caption_len(html2) == 3


@pytest.mark.asyncio
async def test_normalize_titles_fallback(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        raise RuntimeError("no llm")

    monkeypatch.setattr("main.ask_4o", fake_ask)
    titles = ["Лекция Иван Иванов — О языке", "🎨 Лекторий о цвете"]
    res = await normalize_titles_via_4o(titles)
    assert res[0]["emoji"] == ""
    assert res[0]["title_clean"] == "Лекция Иван Иванов: О языке"
    assert res[1]["emoji"] == "🎨"
    assert res[1]["title_clean"] == "о цвете"


@pytest.mark.asyncio
async def test_normalize_titles_masterclass_fallback(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        raise RuntimeError("no llm")

    monkeypatch.setattr("main.ask_4o", fake_ask)
    titles = [
        "Мастер-класс Иван Иванов — Акварель",
        "🎨 Мастер класс Цвет",
    ]
    res = await normalize_titles_via_4o(titles, event_kind="masterclass")
    assert res[0]["emoji"] == ""
    assert res[0]["title_clean"] == "Мастер-класс Иван Иванов: Акварель"
    assert res[1]["emoji"] == "🎨"
    assert res[1]["title_clean"] == "Цвет"


@pytest.mark.asyncio
async def test_normalize_titles_via_llm(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        return (
            '[{"emoji":"📚","title_clean":"Лекция Алёны Мирошниченко: Мода Франции"},'
            '{"emoji":"","title_clean":"Лекция Ильи Дементьева: От каменного века"}]'
        )

    monkeypatch.setattr("main.ask_4o", fake_ask)
    titles = [
        "📚 Лекция Алёны Мирошниченко «Мода Франции»",
        "Лекторий Ильи Дементьева \"От каменного века\"",
    ]
    res = await normalize_titles_via_4o(titles)
    assert res[0]["emoji"] == "📚"
    assert res[0]["title_clean"] == "Лекция Алёны Мирошниченко: Мода Франции"
    assert res[1]["emoji"] == ""
    assert res[1]["title_clean"] == "Лекция Ильи Дементьева: От каменного века"


@pytest.mark.asyncio
async def test_normalize_titles_via_llm_masterclass(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        assert "Мастер-класс" in prompt
        return (
            '[{"emoji":"🎨","title_clean":"Мастер-класс Марии Ивановой: Акварель"},'
            '{"emoji":"","title_clean":"Готовим штрудель"}]'
        )

    monkeypatch.setattr("main.ask_4o", fake_ask)
    titles = [
        "🎨 Мастер-класс Мария Иванова Акварель",
        "Мастер класс Готовим штрудель",
    ]
    res = await normalize_titles_via_4o(titles, event_kind="masterclass")
    assert res[0]["emoji"] == "🎨"
    assert res[0]["title_clean"] == "Мастер-класс Марии Ивановой: Акварель"
    assert res[1]["emoji"] == ""
    assert res[1]["title_clean"] == "Готовим штрудель"


@pytest.mark.asyncio
async def test_normalize_titles_exhibition_keeps_original(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        raise AssertionError("LLM should not be called for exhibitions")

    monkeypatch.setattr("main.ask_4o", fake_ask)
    titles = [
        "🖼️ Выставка — Импрессионисты",
        "Современное искусство без эмодзи",
    ]

    res = await normalize_titles_via_4o(titles, event_kind="exhibition")

    assert res[0]["emoji"] == "🖼️"
    assert res[0]["title_clean"] == "Выставка — Импрессионисты"
    assert res[1]["emoji"] == ""
    assert res[1]["title_clean"] == "Современное искусство без эмодзи"


@pytest.mark.asyncio
async def test_normalize_titles_meetups_adds_exhibition_clarifier(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        raise RuntimeError("no llm")

    monkeypatch.setattr("main.ask_4o", fake_ask)

    events = [
        SimpleNamespace(event_type="выставка", description="Открытие выставки молодых художников."),
    ]
    titles = ["Творческая встреча «Диалог о живописи»"]

    res = await normalize_titles_via_4o(
        titles, event_kind="meetups", events=events
    )

    assert res[0]["emoji"] == ""
    assert res[0]["title_clean"].endswith(
        "— творческая встреча и открытие выставки"
    )


@pytest.mark.asyncio
async def test_normalize_titles_meetups_llm_postprocess(monkeypatch):
    async def fake_ask(prompt, max_tokens=0):
        return '[{"emoji":"","title_clean":"Творческая встреча «Диалог о живописи»"}]'

    monkeypatch.setattr("main.ask_4o", fake_ask)

    events = [
        SimpleNamespace(event_type="выставка", description="Описание говорит о выставке."),
    ]
    titles = ["Творческая встреча «Диалог о живописи»"]

    res = await normalize_titles_via_4o(
        titles, event_kind="meetups", events=events
    )

    assert res[0]["title_clean"].endswith(
        "— творческая встреча и открытие выставки"
    )


def test_format_event_line_and_link_priority():
    e = Event(
        title="T",
        description="d",
        date="2025-05-10",
        time="18:30",
        location_name="L",
        source_text="s",
        event_type="лекция",
        source_post_url="http://t.me/post",
        telegraph_url="http://tg.ph",
    )
    assert pick_display_link(e) == "http://t.me/post"
    line = format_event_line_html(e, pick_display_link(e))
    assert line.startswith('10.05 18:30 | <a href="http://t.me/post">T</a>')

    e.time = "--"  # unparsable
    e.source_post_url = None
    assert pick_display_link(e) == "http://tg.ph"
    line = format_event_line_html(e, pick_display_link(e))
    assert line == '10.05 | <a href="http://tg.ph">T</a>'

    e.telegraph_url = None
    e.telegraph_path = "foo"
    assert pick_display_link(e) == "https://telegra.ph/foo"


def test_format_event_line_html_exhibition_end_date():
    e = Event(
        title="T",
        description="d",
        date="2025-05-10",
        time="18:30",
        location_name="L",
        source_text="s",
        event_type="выставка",
        source_post_url="http://t.me/post",
        end_date="2025-05-12",
    )

    line = format_event_line_html(e, None)

    assert line == "по 12.05 | T"


def test_format_event_line_html_exhibition_missing_end_date(caplog):
    caplog.set_level(logging.WARNING)

    e = Event(
        title="T",
        description="d",
        date="2025-05-10",
        time="18:30",
        location_name="L",
        source_text="s",
        event_type="выставка",
        source_post_url="http://t.me/post",
        end_date=None,
    )

    line = format_event_line_html(e, None)

    assert line == "по 10.05 | T"
    assert any("digest.end_date.missing" in r.message for r in caplog.records)


def test_format_event_line_html_exhibition_bad_end_date(caplog):
    caplog.set_level(logging.WARNING)

    e = Event(
        title="T",
        description="d",
        date="2025-05-10",
        time="18:30",
        location_name="L",
        source_text="s",
        event_type="выставка",
        source_post_url="http://t.me/post",
        end_date="2025/05/12",
    )

    line = format_event_line_html(e, None)

    assert line == "по 10.05 | T"
    assert any("digest.end_date.format" in r.message for r in caplog.records)


def test_aggregate_topics():
    events = [
        SimpleNamespace(topics=["ART", "культура"]),
        SimpleNamespace(topics=["HISTORY_RU", "российская история"]),
        SimpleNamespace(topics=["TECH", "тех"]),
        SimpleNamespace(topics=["MUSIC"]),
        SimpleNamespace(topics=["культура"]),
        SimpleNamespace(topics=["психология"]),
        SimpleNamespace(topics=["mental health"]),
    ]
    assert aggregate_digest_topics(events) == [
        "EXHIBITIONS",
        "PSYCHOLOGY",
        "CONCERTS",
    ]


@pytest.mark.asyncio
async def test_caption_visible_length(caplog):
    intro = "Интро. Второе предложение."  # intro longer than one sentence
    long_url = "http://example.com/" + "a" * 100
    lines = [
        f"01.01 12:00 | <a href=\"{long_url}{i}\">T{i}</a>" for i in range(9)
    ]

    caplog.set_level(logging.INFO)
    caption, used = await assemble_compact_caption(intro, lines, digest_id="x")
    assert visible_caption_len(caption) <= 4096
    assert len(used) == 9
    assert any("digest.caption.compose" in r.message for r in caplog.records)
    assert caption.endswith('\n\n<a href="https://t.me/kenigevents">Полюбить Калининград | Анонсы</a>')


@pytest.mark.asyncio
async def test_compose_caption_excluded(caplog):
    intro = "intro"
    lines = ["l1", "l2", "l3"]
    footer = "<b>f</b>"
    caplog.set_level(logging.INFO)
    caption, used = await compose_digest_caption(
        intro, lines, footer, excluded={1}, digest_id="d1"
    )
    assert used == ["l1", "l3"]
    assert "l2" not in caption
    assert caption.endswith("\n\n<b>f</b>")
    assert visible_caption_len(caption) <= 4096
    assert any("fit_1024" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_candidates_skip_no_link(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2025, 5, 1, 12, 0)

    async with db.get_session() as session:
        for i in range(11):
            dt = now + timedelta(days=i)
            session.add(
                Event(
                    title=f"e{i}",
                    description="d",
                    date=dt.strftime("%Y-%m-%d"),
                    time="15:00",
                    location_name="x",
                    source_text="s",
                    event_type="лекция",
                    source_post_url=None if i == 5 else f"http://example.com/{i}",
                )
            )
        await session.commit()

    events, horizon = await build_lectures_digest_candidates(db, now)
    titles = [e.title for e in events]
    assert len(events) == 9
    assert "e5" not in titles and "e9" in titles


def test_publish_attach_caption_switch():
    media = [types.InputMediaPhoto(media="u1")]
    attach, visible = attach_caption_if_fits(media, "a" * 980)
    assert attach and visible == 980 and media[0].caption
    media2 = [types.InputMediaPhoto(media="u2")]
    attach2, visible2 = attach_caption_if_fits(media2, "b" * 1100)
    assert not attach2 and visible2 == 1100 and media2[0].caption is None
