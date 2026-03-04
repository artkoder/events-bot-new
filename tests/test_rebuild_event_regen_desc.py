import pytest

import main
import smart_event_update as su
from db import Database
from models import Event, EventSource, EventSourceFact, User


@pytest.mark.asyncio
async def test_rebuild_event_regen_desc_updates_description_and_reports(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    try:
        async with db.get_session() as session:
            session.add(User(user_id=123, is_superadmin=True))
            ev = Event(
                title="Хоровая вечеринка",
                description="old",
                date="2026-03-07",
                time="14:00",
                location_name="Октябрьская, 8",
                source_text="src",
            )
            session.add(ev)
            await session.flush()
            src = EventSource(
                event_id=int(ev.id or 0),
                source_type="telegram",
                source_url="https://t.me/kulturnaya_chaika/7375?single",
            )
            session.add(src)
            await session.flush()
            session.add(
                EventSourceFact(
                    event_id=int(ev.id or 0),
                    source_id=int(src.id or 0),
                    fact="Есть плейлист на Я.Музыке: https://music.yandex.ru/users/u/playlists/1030",
                    status="added",
                )
            )
            await session.commit()
            eid = int(ev.id or 0)

        async def _fake_ff_desc(**_kwargs):  # noqa: ANN001
            return (
                "Короткий лид.\n\n"
                "### Музыка\n"
                "Полный список песен — в плейлисте: https://music.yandex.ru/users/u/playlists/1030\n\n"
                "### Участие\n"
                "Можно приходить без подготовки."
            )

        monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)
        monkeypatch.setattr(su, "_llm_fact_first_description_md", _fake_ff_desc)

        async def _fake_schedule_event_update_tasks(_db, _ev, **_kwargs):  # noqa: ANN001
            return {}

        monkeypatch.setattr(main, "schedule_event_update_tasks", _fake_schedule_event_update_tasks)

        class _Bot:
            def __init__(self) -> None:
                self.sent: list[str] = []

            async def send_message(self, _chat_id, text, **_kwargs):  # noqa: ANN001
                self.sent.append(str(text))

        class _Chat:
            id = 999

        class _FromUser:
            id = 123

        class _Message:
            def __init__(self, text: str) -> None:
                self.text = text
                self.chat = _Chat()
                self.from_user = _FromUser()

        bot = _Bot()
        msg = _Message(f"/rebuild_event {eid} --regen-desc")
        await main.handle_rebuild_event_command(msg, db, bot)

        assert bot.sent
        assert "regen_desc=1" in bot.sent[-1]

        async with db.get_session() as session:
            updated = await session.get(Event, eid)
            assert updated is not None
            assert "music.yandex.ru" in (updated.description or "")
    finally:
        await db.close()

