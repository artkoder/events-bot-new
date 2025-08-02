import pytest
import main


@pytest.mark.asyncio
async def test_sync_vk_source_post_adds_comment(monkeypatch):
    main.VK_AFISHA_GROUP_ID = "1"

    event = main.Event(
        title="Title",
        description="",
        date="2024-01-01",
        time="00:00",
        location_name="Place",
    )

    captured_message = {}

    async def fake_post_to_vk(group_id, message, db=None, bot=None, attachments=None, token=None):
        captured_message["text"] = message
        return "https://vk.com/wall-1_2"

    monkeypatch.setattr(main, "post_to_vk", fake_post_to_vk)

    calls = []

    async def fake_vk_api(method, params, db=None, bot=None, token=None):
        calls.append((method, params))
        return {"response": {"comment_id": 1}}

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
    assert calls and calls[0][0] == "wall.createComment"
    assert calls[0][1]["message"] == (
        "Добавить это мероприятие в календарь можно по ссылке: http://ics"
    )
    assert "http://ics" not in captured_message["text"]
    assert captured_message["text"].startswith("[https://source|Title]\nDescription")

