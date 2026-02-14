import pytest

from source_parsing.telegram.handlers import TelegramMonitorReport
from source_parsing.telegram.service import _send_event_details


class _DummyMe:
    username = "eventsbotTestBot"


class _DummyBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict]] = []

    async def get_me(self):
        return _DummyMe()

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((int(chat_id), str(text), kwargs))


@pytest.mark.asyncio
async def test_send_event_details_reports_zero_changes():
    bot = _DummyBot()
    report = TelegramMonitorReport(
        run_id="test-run",
        messages_scanned=0,
        events_extracted=0,
        events_created=0,
        events_merged=0,
    )

    await _send_event_details(bot, 12345, report)

    assert bot.messages, "Expected a Smart Update detail message for zero-change run"
    _chat_id, text, kwargs = bot.messages[-1]
    assert "Smart Update (детали событий)" in text
    assert "Созданные события: 0" in text
    assert "Обновлённые события: 0" in text
    assert kwargs.get("parse_mode") == "HTML"
