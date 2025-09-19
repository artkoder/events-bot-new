import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from main import Database, Event
import logging
from digests import (
    build_lectures_digest_candidates,
    build_masterclasses_digest_candidates,
    build_masterclasses_digest_preview,
    build_exhibitions_digest_candidates,
    build_exhibitions_digest_preview,
    compose_digest_intro_via_4o,
    compose_masterclasses_intro_via_4o,
    compose_exhibitions_intro_via_4o,
    aggregate_digest_topics,
    format_event_line_html,
    pick_display_link,
    normalize_titles_via_4o,
    assemble_compact_caption,
    visible_caption_len,
    compose_digest_caption,
    attach_caption_if_fits,
)
from aiogram import types


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
async def test_build_exhibitions_digest_candidates(tmp_path):
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
                event_type="выставка",
                source_post_url="http://example.com/" + title,
                end_date=(dt + timedelta(days=2)).strftime("%Y-%m-%d"),
            )
            session.add(ev)

        add(0, "15:00", "e0")
        add(5, "12:00", "e5")
        add(9, "12:00", "e9")
        await session.commit()

    events, horizon = await build_exhibitions_digest_candidates(db, now)
    titles = [e.title for e in events]

    assert horizon == 14
    assert titles == ["e0", "e5", "e9"]


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
    assert "выставок" in prompt
    assert "эмодзи" in prompt
    assert "не выдумывай детали" in prompt


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
    )

    captured_payload: dict[str, list] = {}

    async def fake_compose_masterclasses_intro(n, horizon_days, masterclasses):
        captured_payload["value"] = masterclasses
        return "готовое интро"

    async def fake_normalize(titles, *, event_kind="lecture"):
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
    )

    captured_payload: dict[str, list] = {}

    async def fake_compose_exhibitions_intro(n, horizon_days, exhibitions):
        captured_payload["value"] = exhibitions
        return "интро про выставки"

    async def fake_normalize(titles, *, event_kind="lecture"):
        return [{"emoji": "🖼️", "title_clean": "Импрессионисты"} for _ in titles]

    async def fake_candidates(db, now, digest_id=None):
        return [event], 14

    monkeypatch.setattr(
        "digests.compose_exhibitions_intro_via_4o",
        fake_compose_exhibitions_intro,
    )
    monkeypatch.setattr("digests.normalize_titles_via_4o", fake_normalize)
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
            "title": "Импрессионисты",
            "description": "Картины из музеев Европы.",
            "date_range": {"start": "2025-05-02", "end": "2025-05-20"},
        }
    ]
    assert horizon == 14
    assert norm_titles == ["Импрессионисты"]
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

    assert line == "10.05 по 12.05 18:30 | T"


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

    assert line == "10.05 18:30 | T"
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

    assert line == "10.05 18:30 | T"
    assert any("digest.end_date.format" in r.message for r in caplog.records)


def test_aggregate_topics():
    events = [
        SimpleNamespace(topics=["ART", "культура"]),
        SimpleNamespace(topics=["HISTORY_RU", "российская история"]),
        SimpleNamespace(topics=["TECH", "тех"]),
        SimpleNamespace(topics=["MUSIC"]),
        SimpleNamespace(topics=["культура"]),
    ]
    assert aggregate_digest_topics(events) == [
        "EXHIBITIONS",
        "CONCERTS",
        "LECTURES",
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
