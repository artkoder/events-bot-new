import os
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from aiogram import types
from sqlmodel import select

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
from main import Database, User, Event


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=len(self.messages))


@pytest.mark.asyncio
async def test_backfill_topics_command_updates_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        session.add_all(
            [
                Event(
                    title="Event A",
                    description="desc",
                    source_text="src",
                    date=today.isoformat(),
                    time="10:00",
                    location_name="loc",
                    topics=[],
                ),
                Event(
                    title="Event B",
                    description="desc",
                    source_text="src",
                    date=(today + timedelta(days=1)).isoformat(),
                    time="11:00",
                    location_name="loc",
                    topics=["old"],
                ),
                Event(
                    title="Event C",
                    description="desc",
                    source_text="src",
                    date=(today + timedelta(days=2)).isoformat(),
                    time="12:00",
                    location_name="loc",
                    topics=["manual"],
                    topics_manual=True,
                ),
                Event(
                    title="Event D",
                    description="desc",
                    source_text="src",
                    date=(today + timedelta(days=30)).isoformat(),
                    time="13:00",
                    location_name="loc",
                ),
            ]
        )
        await session.commit()

    captured_titles: list[str] = []
    topic_map = {
        "Event A": ["EXHIBITIONS"],
        "Event B": ["CONCERTS"],
    }

    async def fake_classify(event):
        captured_titles.append(event.title)
        return topic_map.get(event.title, [])

    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    message = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 99, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "text": "/backfill_topics 7",
        }
    )
    bot = DummyBot()

    await main.handle_backfill_topics(message, db, bot)

    assert captured_titles == ["Event A", "Event B"]
    assert bot.messages, "Expected summary message to be sent"
    summary_text = bot.messages[-1][1]
    assert "processed=2" in summary_text
    assert "updated=2" in summary_text
    assert "skipped=1" in summary_text

    async with db.get_session() as session:
        stored_a = await session.execute(
            select(Event).where(Event.title == "Event A")
        )
        stored_b = await session.execute(
            select(Event).where(Event.title == "Event B")
        )
        stored_c = await session.execute(
            select(Event).where(Event.title == "Event C")
        )
        stored_d = await session.execute(
            select(Event).where(Event.title == "Event D")
        )
        event_a = stored_a.scalars().first()
        event_b = stored_b.scalars().first()
        event_c = stored_c.scalars().first()
        event_d = stored_d.scalars().first()

    assert event_a.topics == ["EXHIBITIONS"]
    assert event_a.topics_manual is False
    assert event_b.topics == ["CONCERTS"]
    assert event_b.topics_manual is False
    assert event_c.topics == ["manual"]
    assert event_c.topics_manual is True
    assert event_d.topics == []
    assert event_d.topics_manual is False
