import json
from datetime import datetime, timezone, timedelta

import pytest
from aiogram import types
from sqlalchemy import text

import main
from main import Database, Event, User


class DummyBot:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []
        self.documents: list[dict[str, object]] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))
        data = {
            "message_id": len(self.messages),
            "date": 0,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 0, "is_bot": True, "first_name": "B"},
            "text": text,
        }
        return types.Message.model_validate(data)

    async def send_document(self, chat_id, document, **kwargs):
        if isinstance(document, types.BufferedInputFile):
            payload = document.data
            filename = document.filename
        else:
            with open(document.path, "rb") as f:
                payload = f.read()
            filename = document.filename
        self.documents.append({
            "chat_id": chat_id,
            "payload": payload,
            "filename": filename,
        })
        data = {
            "message_id": len(self.messages) + len(self.documents),
            "date": 0,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 0, "is_bot": True, "first_name": "B"},
            "document": {
                "file_id": "1",
                "file_unique_id": "1",
                "file_name": filename,
                "file_size": len(payload),
            },
        }
        return types.Message.model_validate(data)


@pytest.mark.asyncio
async def test_tourist_export_collects_all_events(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        await session.execute(text("ALTER TABLE event ADD COLUMN tourist_event_type TEXT"))
        await session.execute(text("ALTER TABLE event ADD COLUMN tourist_tags TEXT"))
        user = User(user_id=1, username="mod", is_superadmin=False)
        session.add(user)
        today = datetime.now(timezone.utc).date()
        ev1 = Event(
            title="Event 1",
            description="Desc 1",
            date=(today + timedelta(days=1)).isoformat(),
            time="12:00",
            location_name="Loc 1",
            city="City",
            source_text="src1",
        )
        ev2 = Event(
            title="Event 2",
            description="Desc 2",
            date=(today + timedelta(days=10)).isoformat(),
            end_date=(today + timedelta(days=11)).isoformat(),
            time="18:00",
            location_name="Loc 2",
            city="City",
            source_text="src2",
        )
        ev3 = Event(
            title="Event 3",
            description="Desc 3",
            date=(today - timedelta(days=30)).isoformat(),
            time="10:00",
            location_name="Loc 3",
            city="City",
            source_text="src3",
        )
        session.add_all([ev1, ev2, ev3])
        await session.commit()
        await session.execute(
            text(
                "UPDATE event SET tourist_event_type = :type, tourist_tags = :tags WHERE id = :id"
            ),
            {"type": "family", "tags": '["museum"]', "id": ev1.id},
        )
        await session.execute(
            text(
                "UPDATE event SET tourist_event_type = :type, tourist_tags = :tags WHERE id = :id"
            ),
            {"type": "music", "tags": '["concert"]', "id": ev2.id},
        )
        await session.commit()

    cmd = "/tourist_export period=" + (
        f"{(today - timedelta(days=5)).isoformat()}..{(today + timedelta(days=20)).isoformat()}"
    )
    message = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": cmd,
        }
    )
    bot = DummyBot()

    await main.handle_tourist_export(message, db, bot)

    assert not bot.messages, "command should send document without extra messages"
    assert bot.documents, "document with export should be sent"
    export_bytes = bot.documents[0]["payload"]
    lines = [line for line in export_bytes.decode("utf-8").splitlines() if line]
    assert len(lines) == 2
    exported = [json.loads(line) for line in lines]
    exported_ids = {item["id"] for item in exported}
    assert exported_ids == {1, 2}
    for item in exported:
        assert "tourist_event_type" in item
        assert "tourist_tags" in item
        assert isinstance(item["tourist_tags"], list)
        if item["id"] == 1:
            assert item["tourist_event_type"] == "family"
            assert item["tourist_tags"] == ["museum"]
        if item["id"] == 2:
            assert item["tourist_event_type"] == "music"
            assert item["tourist_tags"] == ["concert"]
