import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from main import Database, Event
import logging
from digests import (
    build_lectures_digest_candidates,
    compose_digest_intro_via_4o,
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
        return "интро"

    monkeypatch.setattr("main.ask_4o", fake_ask)
    text = await compose_digest_intro_via_4o(2, 7, ["e1", "e2"])
    assert text == "интро"
    assert any("digest.intro.llm.request" in r.message for r in caplog.records)
    assert any("digest.intro.llm.response" in r.message for r in caplog.records)


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


def test_aggregate_topics():
    events = [
        SimpleNamespace(topics=["искусство", "культура"]),
        SimpleNamespace(topics=["история россии", "российская история"]),
        SimpleNamespace(topics=["ит", "тех"]),
        SimpleNamespace(topics=["музыка"]),
        SimpleNamespace(topics=["культура"]),
    ]
    assert aggregate_digest_topics(events) == [
        "искусство",
        "история россии",
        "музыка",
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
