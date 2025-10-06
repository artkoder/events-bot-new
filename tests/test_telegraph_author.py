import importlib
from aiogram import types
import pytest
import main as orig_main


@pytest.mark.asyncio
async def test_telegraph_wrappers_add_author(monkeypatch):
    m = importlib.reload(orig_main)

    async def fake_call(func, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(m, "telegraph_call", fake_call)

    class DummyTelegraph:
        def create_page(self, *args, **kwargs):
            raise AssertionError("should not be called")

        def edit_page(self, *args, **kwargs):
            raise AssertionError("should not be called")

    tg = DummyTelegraph()
    res = await m.telegraph_create_page(tg, title="T")
    assert res["kwargs"]["author_name"] == m.TELEGRAPH_AUTHOR_NAME
    assert res["kwargs"]["author_url"] == m.TELEGRAPH_AUTHOR_URL

    res2 = await m.telegraph_edit_page(tg, "p", title="T")
    assert res2["kwargs"]["author_name"] == m.TELEGRAPH_AUTHOR_NAME
    assert res2["kwargs"]["author_url"] == m.TELEGRAPH_AUTHOR_URL

    res3 = await m.telegraph_create_page(
        tg, title="T", author_name="X", author_url="Y"
    )
    assert res3["kwargs"]["author_name"] == "X"
    assert res3["kwargs"]["author_url"] == "Y"


@pytest.mark.asyncio
async def test_create_source_page_history_author_url(monkeypatch):
    m = importlib.reload(orig_main)

    async def fake_build_source_page_content(*args, **kwargs):
        return "<p>content</p>", "", 0

    monkeypatch.setattr(m, "build_source_page_content", fake_build_source_page_content)
    monkeypatch.setattr(m, "get_telegraph_token", lambda: "token")
    monkeypatch.setattr(m, "normalize_telegraph_url", lambda url: url)

    class DummyTelegraph:
        pass

    monkeypatch.setattr(m, "Telegraph", lambda access_token: DummyTelegraph())

    async def fake_create_page(tg, **kwargs):
        fake_create_page.calls.append(kwargs)
        return {"url": "https://telegra.ph/test", "path": "test"}

    fake_create_page.calls = []
    monkeypatch.setattr(m, "telegraph_create_page", fake_create_page)

    import telegraph.utils as telegraph_utils

    def fake_html_to_nodes(html):
        return [html]

    class DummyInvalidHTML(Exception):
        pass

    monkeypatch.setattr(telegraph_utils, "html_to_nodes", fake_html_to_nodes)
    monkeypatch.setattr(telegraph_utils, "InvalidHTML", DummyInvalidHTML)

    await m.create_source_page(
        "Title",
        "text",
        None,
        page_mode="history",
    )

    assert fake_create_page.calls
    assert (
        fake_create_page.calls[0]["author_url"] == m.HISTORY_TELEGRAPH_AUTHOR_URL
    )


@pytest.mark.asyncio
async def test_telegraph_fix_author(monkeypatch, tmp_path):
    import main

    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            main.Event(
                title="E",
                description="d",
                date="2024-01-01",
                time="00:00",
                location_name="L",
                source_text="s",
                telegraph_path="p1",
            )
        )
        session.add(
            main.MonthPage(month="2024-01", url="u", path="mp1", path2="mp2")
        )
        session.add(
            main.WeekendPage(start="2024-01-05", url="u", path="wp1")
        )
        session.add(main.Festival(name="Fest", telegraph_path="fp1"))
        await session.commit()

    calls = []

    async def fake_edit_page(tg, path, **kwargs):
        calls.append(path)

    monkeypatch.setattr(main, "telegraph_edit_page", fake_edit_page)

    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(main.random, "uniform", lambda a, b: 0.7)

    class Bot:
        def __init__(self):
            self.messages = []

        async def send_message(self, chat_id, text):
            self.messages.append((chat_id, text))

    bot = Bot()
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/telegraph_fix_author",
        }
    )

    await main.handle_telegraph_fix_author(msg, db, bot)

    assert len(calls) == 5
    assert all(abs(d - 0.7) < 1e-6 for d in sleeps)
    assert any("Готово" in m[1] for m in bot.messages)
