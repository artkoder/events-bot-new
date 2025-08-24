import pytest
from aiogram import Bot, types
import main
from main import Database, Event, Channel


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.docs = []

    async def send_document(self, chat_id, document, caption=None, **kwargs):
        self.docs.append((chat_id, caption))
        from types import SimpleNamespace
        return SimpleNamespace(document=SimpleNamespace(file_id="file" + str(len(self.docs))), message_id=len(self.docs))


class FakeClient:
    def __init__(self):
        self.storage = self

    def from_(self, bucket):
        self.bucket = bucket
        return self

    def upload(self, path, content, options):
        pass

    def get_public_url(self, path):
        return f"https://supabase/{path}"


@pytest.mark.asyncio
async def test_ics_fields_persist_and_update(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    await main.ics_publish(1, db, bot)
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        h1, u1, f1, t1 = ev.ics_hash, ev.ics_url_supabase, ev.ics_file_id, ev.ics_updated_at
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        ev.time = "20:00"
        await session.commit()
    await main.ics_publish(1, db, bot)
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.ics_hash != h1
        assert ev.ics_url_supabase
        assert ev.ics_file_id != f1
        assert ev.ics_updated_at > t1
