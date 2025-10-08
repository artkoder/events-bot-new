import re
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

        def is_(self, column, value):
            state["is_"] = (column, value)
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
    assert "checked" in state["select"][0]
    assert state["is_"] == ("checked", False)
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
    published_at = datetime(2024, 5, 2, 15, 30, tzinfo=timezone.utc)
    await main._vk_miss_append_feedback(record, "  sample text  ", published_at)

    content = path.read_text(encoding="utf-8")
    assert "sample text" in content
    assert "https://vk.com/wall-1_2" in content
    assert "miss" in content
    expected_recorded = main._vk_miss_format_timestamp(record.timestamp)
    assert (
        f"- Время фиксации пропуска: {expected_recorded} (время попадания в отклонённые)"
        in content
    )
    assert "Дата публикации" in content


@pytest.mark.asyncio
async def test_fetch_vk_post_preview_returns_date(monkeypatch):
    ts = 1_700_000_000

    async def fake_wall_get_items(group_id, post_id, db, bot):
        return [
            {
                "text": "Sample post",
                "date": ts,
                "attachments": [],
            }
        ]

    def fake_extract_photo_urls(items):
        return ["https://example.com/photo.jpg"]

    monkeypatch.setattr(main, "_vk_wall_get_items", fake_wall_get_items)
    monkeypatch.setattr(main, "_vk_extract_photo_urls", fake_extract_photo_urls)

    text, photos, published_at = await main.fetch_vk_post_preview(1, 2, None, None)

    assert text == "Sample post"
    assert photos == ["https://example.com/photo.jpg"]
    assert isinstance(published_at, datetime)
    assert published_at.tzinfo is not None
    assert published_at == datetime.fromtimestamp(ts, tz=timezone.utc)


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
    session.last_published_at = datetime.now(timezone.utc)
    first_published = session.last_published_at
    main.vk_miss_review_sessions[user_id] = session

    calls: list[tuple] = []

    async def fake_show_next(uid, chat_id, db, bot):
        calls.append(("show", uid, chat_id))

    async def fake_append(record, text, published_at=None):
        calls.append(("append", record.id, text, published_at))

    async def fake_mark(record_id):
        calls.append(("mark", record_id))

    monkeypatch.setattr(main, "_vk_miss_show_next", fake_show_next)
    monkeypatch.setattr(main, "_vk_miss_append_feedback", fake_append)
    monkeypatch.setattr(main, "_vk_miss_mark_checked", fake_mark)

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
    assert session.last_published_at is None
    appended = [c for c in calls if c[0] == "append"]
    assert appended and appended[0][1:] == (record1.id, "text1", first_published)
    assert ("mark", record1.id) in calls

    calls.clear()
    session.last_text = "text2"
    session.last_published_at = datetime.now(timezone.utc)
    callback_ok = DummyCallback("vkmiss:ok:1")
    await main.handle_vk_miss_review_callback(callback_ok, None, None)

    assert session.index == 2
    assert all(call[0] != "append" for call in calls)
    assert ("show", user_id, 777) in calls
    assert ("mark", record2.id) in calls

    main.vk_miss_review_sessions.pop(user_id, None)


@pytest.mark.asyncio
async def test_vk_miss_show_next_offers_feedback_file(monkeypatch):
    user_id = 99
    chat_id = 555
    main.vk_miss_review_sessions[user_id] = main.VkMissReviewSession(queue=[], index=0)

    calls: list[tuple] = []

    class DummyBot:
        async def send_message(self, chat_id, text, **kwargs):
            calls.append(("msg", chat_id, text))

    async def fake_offer(bot, chat_id):
        calls.append(("offer", chat_id))

    monkeypatch.setattr(main, "_vk_miss_offer_feedback_file", fake_offer)

    await main._vk_miss_show_next(user_id, chat_id, None, DummyBot())

    assert any(call[0] == "offer" for call in calls)
    assert main.vk_miss_review_sessions.get(user_id) is None


@pytest.mark.asyncio
async def test_vk_miss_download_sends_and_clears(tmp_path, monkeypatch):
    path = tmp_path / "vk_miss.md"
    path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(main, "VK_MISS_REVIEW_FILE", str(path))

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)

    documents: list[tuple[int, object]] = []
    answers: list[tuple[str | None, bool]] = []

    class DummyBot:
        async def send_document(self, chat_id, document, **kwargs):
            documents.append((chat_id, document))

    class DummyMessage:
        def __init__(self):
            self.chat = types.SimpleNamespace(id=111)

    class DummyCallback:
        def __init__(self):
            self.data = "vkmiss:download"
            self.from_user = types.SimpleNamespace(id=222)
            self.message = DummyMessage()

        async def answer(self, text: str | None = None, show_alert: bool = False):
            answers.append((text, show_alert))

    callback = DummyCallback()
    await main.handle_vk_miss_review_callback(callback, None, DummyBot())

    assert documents and documents[0][0] == 111
    assert isinstance(documents[0][1], main.types.FSInputFile)
    filename = documents[0][1].filename
    assert filename.startswith("vk_miss_review_")
    assert re.search(r"^vk_miss_review_\d{8}-\d{6}\.md$", filename)
    assert answers and answers[-1][0] == "Файл отправлен"
    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_vk_miss_download_alerts_on_empty(tmp_path, monkeypatch):
    path = tmp_path / "vk_miss_empty.md"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(main, "VK_MISS_REVIEW_FILE", str(path))

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)

    answers: list[tuple[str | None, bool]] = []

    class DummyBot:
        async def send_document(self, *args, **kwargs):  # pragma: no cover - not used
            raise AssertionError("send_document should not be called")

    class DummyCallback:
        def __init__(self):
            self.data = "vkmiss:download"
            self.from_user = types.SimpleNamespace(id=333)
            self.message = types.SimpleNamespace(chat=types.SimpleNamespace(id=444))

        async def answer(self, text: str | None = None, show_alert: bool = False):
            answers.append((text, show_alert))

    callback = DummyCallback()
    await main.handle_vk_miss_review_callback(callback, None, DummyBot())

    assert answers == [("Файл отсутствует или пуст", True)]
    assert path.read_text(encoding="utf-8") == ""
