import types
from datetime import datetime, timezone

import pytest

import main


@pytest.mark.asyncio
async def test_fetch_vk_miss_samples(monkeypatch):
    rows = [
        {
            "id": "-123_456",
            "url": "https://vk.com/wall-123_456",
            "reason": "late",
            "matched_kw": "concert",
            "ts": "2024-05-02T12:34:56+00:00",
        }
    ]
    state: dict[str, object] = {}

    class FakeQuery:
        def __init__(self, data):
            self._data = data

        def select(self, *_):
            state["select"] = _
            return self

        def order(self, column, *, desc=False):
            state["order"] = (column, desc)
            return self

        def limit(self, value):
            state["limit"] = value
            return self

        def execute(self):
            return types.SimpleNamespace(data=self._data)

    class FakeClient:
        def table(self, name):
            state["table"] = name
            return FakeQuery(rows)

    monkeypatch.setattr(main, "get_supabase_client", lambda: FakeClient())

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)

    records = await main.fetch_vk_miss_samples(5)

    assert len(records) == 1
    record = records[0]
    assert record.id == "-123_456"
    assert record.url == "https://vk.com/wall-123_456"
    assert record.reason == "late"
    assert record.matched_kw == "concert"
    assert record.timestamp.tzinfo is not None
    assert state["table"] == "vk_misses_sample"
    assert state["limit"] == 5
    assert state["order"] == ("ts", True)


@pytest.mark.asyncio
async def test_vk_miss_append_feedback_writes_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "miss.md"
    monkeypatch.setattr(main, "VK_MISS_REVIEW_FILE", str(path))

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)

    record = main.VkMissRecord(
        id="-1_2",
        url="https://vk.com/wall-1_2",
        reason="miss",
        matched_kw=None,
        timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    await main._vk_miss_append_feedback(record, "  sample text  ")

    content = path.read_text(encoding="utf-8")
    assert "sample text" in content
    assert "https://vk.com/wall-1_2" in content
    assert "miss" in content


@pytest.mark.asyncio
async def test_vk_miss_callbacks_progress(monkeypatch):
    user_id = 42
    record1 = main.VkMissRecord(
        id="-1_1",
        url="https://vk.com/wall-1_1",
        reason="old",
        matched_kw="kw1",
        timestamp=datetime.now(timezone.utc),
    )
    record2 = main.VkMissRecord(
        id="-1_2",
        url="https://vk.com/wall-1_2",
        reason="late",
        matched_kw="kw2",
        timestamp=datetime.now(timezone.utc),
    )
    session = main.VkMissReviewSession(queue=[record1, record2], index=0, last_text="text1")
    main.vk_miss_review_sessions[user_id] = session

    calls: list[tuple] = []

    async def fake_show_next(uid, chat_id, db, bot):
        calls.append(("show", uid, chat_id))

    async def fake_append(record, text):
        calls.append(("append", record.id, text))

    monkeypatch.setattr(main, "_vk_miss_show_next", fake_show_next)
    monkeypatch.setattr(main, "_vk_miss_append_feedback", fake_append)

    class DummyMessage:
        def __init__(self):
            self.chat = types.SimpleNamespace(id=777)

    class DummyCallback:
        def __init__(self, data: str):
            self.data = data
            self.from_user = types.SimpleNamespace(id=user_id)
            self.message = DummyMessage()

        async def answer(self, text: str | None = None, show_alert: bool = False):
            calls.append(("answer", text, show_alert))

    callback = DummyCallback("vkmiss:redo:0")
    await main.handle_vk_miss_review_callback(callback, None, None)

    assert session.index == 1
    assert session.last_text is None
    assert ("append", record1.id, "text1") in calls

    calls.clear()
    session.last_text = "text2"
    callback_ok = DummyCallback("vkmiss:ok:1")
    await main.handle_vk_miss_review_callback(callback_ok, None, None)

    assert session.index == 2
    assert ("append", record1.id, "text1") not in calls
    assert ("show", user_id, 777) in calls

    main.vk_miss_review_sessions.pop(user_id, None)
