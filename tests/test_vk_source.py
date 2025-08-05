import pytest
from pathlib import Path
import main


@pytest.mark.asyncio
async def test_sync_vk_source_post_includes_calendar_link(monkeypatch):
    main.VK_AFISHA_GROUP_ID = "1"

    event = main.Event(
        title="Title",
        description="",
        date="2024-01-01",
        time="00:00",
        location_name="Place",
    )

    captured_message = {}

    async def fake_post_to_vk(
        group_id, message, db=None, bot=None, attachments=None, token=None
    ):
        captured_message["text"] = message
        return "https://vk.com/wall-1_2"

    monkeypatch.setattr(main, "post_to_vk", fake_post_to_vk)

    calls = []

    async def fake_vk_api(method, params, db=None, bot=None, token=None):
        calls.append((method, params))
        return {"response": {}}

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)

    url = await main.sync_vk_source_post(
        event,
        "Title\nDescription",
        None,
        None,
        ics_url="http://ics",
    )

    assert url == "https://vk.com/wall-1_2"
    assert not any(method == "wall.createComment" for method, _ in calls)
    lines = captured_message["text"].splitlines()
    assert lines[0] == "Title"
    assert lines[1] == main.VK_BLANK_LINE
    assert "Добавить в календарь http://ics" in captured_message["text"]
    assert captured_message["text"].endswith(main.VK_SOURCE_FOOTER)


def test_build_vk_source_message_converts_links():
    text = "Регистрация [здесь](http://reg) и <a href=\"http://pay\">билеты</a>"
    event = main.Event(
        title="T",
        description="",
        date="2024-01-01",
        time="00:00",
        location_name="Place",
    )
    msg = main.build_vk_source_message(event, text)
    assert "здесь (http://reg)" in msg
    assert "билеты (http://pay)" in msg


@pytest.mark.asyncio
async def test_add_events_from_text_preserves_links(tmp_path: Path, monkeypatch):
    main.VK_AFISHA_GROUP_ID = ""
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_parse(text: str, source_channel: str | None = None, festival_names=None):
        return [
            {
                "title": "T",
                "short_description": "d",
                "date": "2099-01-01",
                "time": "18:00",
                "location_name": "Hall",
            }
        ]

    async def fake_create(title, text, source, html_text=None, media=None, ics_url=None, db=None, **kwargs):
        return "url", "p"

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(main, "create_source_page", fake_create)

    html = "<a href='http://reg'>Регистрация</a>"
    res = await main.add_events_from_text(db, "Регистрация", None, html, None)
    ev = res[0][0]
    assert "Регистрация (http://reg)" in ev.source_text


@pytest.mark.asyncio
async def test_sync_vk_source_post_appends_only_text(monkeypatch):
    main.VK_AFISHA_GROUP_ID = "1"
    event = main.Event(
        title="Old",
        description="",
        date="2024-01-01",
        time="00:00",
        location_name="Old Place",
    )
    event.source_vk_post_url = "https://vk.com/wall-1_1"

    existing = main.build_vk_source_message(event, "old text")

    async def fake_vk_api(method, params, db=None, bot=None, token=None):
        return {"response": [{"text": existing}]}

    edited = {}

    async def fake_edit(url, message, db=None, bot=None):
        edited["text"] = message

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)

    event.title = "New"
    event.location_name = "New Place"

    url = await main.sync_vk_source_post(
        event, "new text", None, None, append_text=True
    )

    assert url == "https://vk.com/wall-1_1"
    msg = edited["text"]
    lines = msg.splitlines()
    assert lines[0] == "New"
    assert "Old Place" not in msg
    assert "old text" in msg
    assert "new text" in msg
    assert msg.count(main.CONTENT_SEPARATOR) == 1
    assert lines.count("New") == 1


@pytest.mark.asyncio
async def test_sync_vk_source_post_updates_without_append(monkeypatch):
    main.VK_AFISHA_GROUP_ID = "1"
    event = main.Event(
        title="Old Title",
        description="",
        date="2024-01-01",
        time="00:00",
        location_name="Place",
    )
    event.source_vk_post_url = "https://vk.com/wall-1_1"

    existing = main.build_vk_source_message(event, "old text")

    async def fake_vk_api(method, params, db=None, bot=None, token=None):
        return {"response": [{"text": existing}]}

    edited = {}

    async def fake_edit(url, message, db=None, bot=None):
        edited["text"] = message

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    monkeypatch.setattr(main, "edit_vk_post", fake_edit)

    event.title = "Updated Title"

    url = await main.sync_vk_source_post(
        event, "updated text", None, None, append_text=False
    )

    assert url == "https://vk.com/wall-1_1"
    msg = edited["text"]
    lines = msg.splitlines()
    assert lines[0] == "Updated Title"
    assert "old text" not in msg
    assert "updated text" in msg
    assert main.CONTENT_SEPARATOR not in msg

