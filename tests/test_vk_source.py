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
        "https://source",
        "Title\nDescription",
        None,
        None,
        ics_url="http://ics",
    )

    assert url == "https://vk.com/wall-1_2"
    assert not any(method == "wall.createComment" for method, _ in calls)
    assert "Добавить в календарь http://ics" in captured_message["text"]
    assert captured_message["text"].startswith("[https://source|Title]\nDescription")


def test_build_vk_source_message_converts_links():
    text = "Регистрация [здесь](http://reg) и <a href=\"http://pay\">билеты</a>"
    msg = main.build_vk_source_message(text, None, display_link=False)
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

