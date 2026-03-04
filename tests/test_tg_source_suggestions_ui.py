from __future__ import annotations

import pytest

import main
from models import TelegramSource
from source_parsing.telegram import commands as tg_commands


class _DummyMessage:
    def __init__(self) -> None:
        self.answers: list[str] = []

    async def answer(self, text: str, **_kwargs):
        self.answers.append(str(text))


@pytest.mark.asyncio
async def test_accept_source_festival_suggestion_sets_manual_value(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        src = TelegramSource(
            username="testchan",
            enabled=True,
            suggested_festival_series="Открытое море",
        )
        session.add(src)
        await session.commit()
        await session.refresh(src)
        source_id = int(src.id)

    async def _noop_list_sources(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tg_commands, "list_sources", _noop_list_sources)

    msg = _DummyMessage()
    await tg_commands.accept_source_festival_suggestion(db, msg, source_id, page=1)

    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        assert src is not None
        assert src.festival_series == "Открытое море"
        assert bool(src.festival_source) is True

    assert any("Принята подсказка" in text for text in msg.answers)
