import pytest
from aiogram import Bot

import main
from main import Database, Event, MonthPage, update_source_post_keyboard


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.edits = []

    async def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None, **kwargs):
        self.edits.append((chat_id, message_id, reply_markup))
        from types import SimpleNamespace
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_update_keyboard_with_ics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            MonthPage(month="2025-08", url="https://t.me/c/m2", path="p2"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                ics_post_url="https://t.me/c/asset/1",
                source_chat_id=-100,
                source_message_id=10,
            ),
        ])
        await session.commit()
    await update_source_post_keyboard(1, db, bot)
    assert bot.edits
    markup = bot.edits[0][2]
    assert len(markup.inline_keyboard) == 2
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/asset/1"
    assert "Добавить в календарь" == markup.inline_keyboard[0][0].text
    assert len(markup.inline_keyboard[1]) == 2
    assert markup.inline_keyboard[1][0].url == "https://t.me/c/m1"
    assert markup.inline_keyboard[1][1].url == "https://t.me/c/m2"


@pytest.mark.asyncio
async def test_update_keyboard_without_ics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add_all([
            MonthPage(month="2025-07", url="https://t.me/c/m1", path="p1"),
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
                source_chat_id=-100,
                source_message_id=10,
            ),
        ])
        await session.commit()
    await update_source_post_keyboard(1, db, bot)
    assert bot.edits
    markup = bot.edits[0][2]
    assert len(markup.inline_keyboard) == 1
    assert len(markup.inline_keyboard[0]) == 1
    assert markup.inline_keyboard[0][0].url == "https://t.me/c/m1"
