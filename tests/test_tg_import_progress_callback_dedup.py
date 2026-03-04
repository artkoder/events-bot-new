import pytest


class DummyBot:
    def __init__(self) -> None:
        self.sent = []
        self.edited = []
        self._next_id = 1000

    async def send_message(self, chat_id: int, text: str, **kwargs):
        mid = self._next_id
        self._next_id += 1
        self.sent.append((int(chat_id), str(text), dict(kwargs), int(mid)))

        # Mimic aiogram message object minimally.
        from types import SimpleNamespace

        return SimpleNamespace(
            message_id=int(mid),
            chat=SimpleNamespace(id=int(chat_id)),
            text=str(text),
            reply_markup=None,
        )

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, **kwargs):
        self.edited.append((int(chat_id), int(message_id), str(text), dict(kwargs)))
        return None


@pytest.mark.asyncio
async def test_tg_import_progress_start_is_upserted_not_resent():
    from source_parsing.telegram.handlers import TelegramMonitorImportProgress
    from source_parsing.telegram.service import _make_import_progress_callback

    bot = DummyBot()
    cb = _make_import_progress_callback(db=object(), bot=bot, chat_id=123, send_progress=True)

    p = TelegramMonitorImportProgress(
        stage="start",
        status="running",
        current_no=1,
        total_no=10,
        source_username="testchannel",
        source_title="Test Channel",
        message_id=111,
        source_link="https://t.me/testchannel/111",
        events_extracted=1,
        events_imported=0,
    )

    await cb(p)
    await cb(p)  # duplicate start should edit the same message, not send a new one

    assert len(bot.sent) == 1
    assert len(bot.edited) >= 1


@pytest.mark.asyncio
async def test_tg_import_progress_done_is_deduped():
    from source_parsing.telegram.handlers import TelegramMonitorImportProgress
    from source_parsing.telegram.service import _make_import_progress_callback

    bot = DummyBot()
    cb = _make_import_progress_callback(db=object(), bot=bot, chat_id=123, send_progress=True)

    start = TelegramMonitorImportProgress(
        stage="start",
        status="running",
        current_no=1,
        total_no=10,
        source_username="testchannel",
        source_title="Test Channel",
        message_id=111,
        source_link="https://t.me/testchannel/111",
        events_extracted=0,
        events_imported=0,
    )
    done = TelegramMonitorImportProgress(
        stage="done",
        status="skipped",
        current_no=1,
        total_no=10,
        source_username="testchannel",
        source_title="Test Channel",
        message_id=111,
        source_link="https://t.me/testchannel/111",
        events_extracted=0,
        events_imported=0,
        reason="past_event",
        skip_breakdown={"past_event": 1},
        took_sec=1.2,
    )

    await cb(start)
    await cb(done)
    before = (len(bot.sent), len(bot.edited))
    await cb(done)  # duplicate done should be ignored
    after = (len(bot.sent), len(bot.edited))

    assert after == before
