from __future__ import annotations

from pathlib import Path

import pytest
from aiogram import types

import main
from db import Database
from festival_activities import (
    FestivalActivitiesError,
    activities_to_nodes,
    format_activity_slot,
    group_activities_by_city,
    normalize_venues,
    parse_festival_activities_yaml,
)
from models import Festival, User


def test_parse_festival_activities_yaml_basic():
    yaml_text = """
    version: 2
    defaults:
      city: Калининград
    festival:
      website: https://fest.example
    activities:
      - title: Лекция
        kind: Лекции
        summary: Описание
        venues:
          - Музей Янтаря
        schedule:
          - date: 2024-06-01
            start: "10:00"
            end: "12:00"
            price: 500₽
            cta:
              text: Регистрация
              url: https://tickets
    """

    result = parse_festival_activities_yaml(yaml_text)
    assert result.website_url == "https://fest.example"
    assert len(result.activities) == 1
    activity = result.activities[0]
    assert activity["title"] == "Лекция"
    assert activity["kind"] == "Лекции"
    venue = activity["venues"][0]
    assert venue["name"].startswith("Музей Янтаря")
    assert venue["city"] == "Калининград"
    slot = activity["schedule"][0]
    assert slot["start_time"] == "10:00"
    assert slot["cta_text"] == "Регистрация"


def test_normalize_venues_prefers_default_city():
    venues = normalize_venues(
        [{"name": "Пространство", "address": "Улица 1"}],
        default_city="Советск",
        canonical={},
    )
    assert venues[0]["city"] == "Советск"


def test_group_activities_by_city_order():
    items = [
        {"title": "A", "kind": "Лекции", "venues": [{"city": "Калининград"}]},
        {"title": "B", "kind": "Лекции", "venues": [{"city": "Светлогорск"}]},
        {"title": "C", "kind": "Лекции", "venues": [{"city": None}]},
    ]
    groups = group_activities_by_city(items)
    assert [city for city, _ in groups] == ["Калининград", "Светлогорск", None]


def test_format_activity_slot_text():
    slot = {
        "date": "2024-07-02",
        "start_time": "11:00",
        "end_time": "13:00",
        "label": "Мастер-класс",
        "price": "700₽",
        "notes": "Для всей семьи",
    }
    text = format_activity_slot(slot)
    assert "2 июля" in text
    assert "11:00–13:00" in text
    assert "700₽" in text


def test_activities_to_nodes_creates_sections():
    activities = [
        {
            "title": "Anytime",
            "kind": "Экспозиции",
            "summary": None,
            "description": None,
            "anytime": True,
            "on_request": False,
            "tags": [],
            "venues": [{"name": "Зал", "address": None, "city": "Калининград"}],
            "schedule": [],
            "cta_text": None,
            "cta_url": None,
        },
        {
            "title": "Request",
            "kind": "Экскурсии",
            "summary": None,
            "description": None,
            "anytime": False,
            "on_request": True,
            "tags": [],
            "venues": [{"name": "Парк", "address": None, "city": "Зеленоградск"}],
            "schedule": [],
            "cta_text": None,
            "cta_url": None,
        },
        {
            "title": "Talk",
            "kind": "Лекции",
            "summary": None,
            "description": None,
            "anytime": False,
            "on_request": False,
            "tags": [],
            "venues": [{"name": "Библиотека", "address": None, "city": "Калининград"}],
            "schedule": [
                {
                    "date": "2024-07-04",
                    "start_time": "15:00",
                    "end_time": None,
                    "label": None,
                    "notes": None,
                    "price": None,
                    "cta_text": None,
                    "cta_url": None,
                }
            ],
            "cta_text": None,
            "cta_url": None,
        },
    ]

    nodes = activities_to_nodes(activities)
    headings = [
        n["children"][0]
        for n in nodes
        if isinstance(n, dict) and n.get("tag") in {"h2", "h3"}
    ]
    assert headings[:3] == ["АКТИВНОСТИ", "Можно в любой день", "По запросу / по записи"]
    assert "Лекции" in headings


@pytest.mark.asyncio
async def test_festival_activities_admin_flow(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "sync_festival_page", noop)
    monkeypatch.setattr(main, "sync_festival_vk_post", noop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", noop)

    async with db.get_session() as session:
        session.add(User(user_id=1))
        fest = Festival(name="Fest")
        session.add(fest)
        await session.commit()
        fid = fest.id

    yaml_payload = """
    version: 2
    festival:
      website: https://fest.example
    activities:
      - title: Маршрут
        kind: Экскурсии
        venues:
          - Дом китобоя
        schedule:
          - date: 2024-07-01
            start: "09:00"
    """

    class DummyBot:
        def __init__(self):
            self.sent: list[tuple[int, str]] = []

        async def send_message(self, chat_id, text, **kwargs):
            self.sent.append((chat_id, text))

        async def download(self, *args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("download should not be used in this test")

    bot = DummyBot()

    main.festival_edit_sessions.pop(1, None)
    main.festival_edit_sessions[1] = (fid, main.FESTIVAL_EDIT_FIELD_ACTIVITIES)

    message = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "text": yaml_payload,
        }
    )

    await main.handle_festival_edit_message(message, db, bot)  # type: ignore[arg-type]

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest is not None
        assert fest.website_url == "https://fest.example"
        assert len(fest.activities_json) == 1

    assert main.festival_edit_sessions[1] == (fid, None)
    preview_messages = [text for _, text in bot.sent if "Предпросмотр" in text]
    assert preview_messages, "admin flow should send preview"

    main.festival_edit_sessions.pop(1, None)


@pytest.mark.asyncio
async def test_festacts_callback_sets_state(tmp_path: Path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(User(user_id=1))
        fest = Festival(
            name="Fest",
            activities_json=[
                {
                    "title": "Any",
                    "kind": "Лекции",
                    "summary": None,
                    "description": None,
                    "anytime": True,
                    "on_request": False,
                    "tags": [],
                    "venues": [
                        {"name": "Дом китобоя", "address": None, "city": "Калининград"}
                    ],
                    "schedule": [],
                    "cta_text": None,
                    "cta_url": None,
                }
            ],
        )
        session.add(fest)
        await session.commit()
        fid = fest.id

    class DummyBot:
        def __init__(self):
            self.messages: list[str] = []

        async def send_message(self, chat_id, text, **kwargs):
            self.messages.append(text)

    bot = DummyBot()

    main.festival_edit_sessions.pop(1, None)
    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "data": f"festacts:{fid}",
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "chat_instance": "1",
            "message": {
                "message_id": 1,
                "date": 0,
                "chat": {"id": 1, "type": "private"},
                "from": {"id": 999, "is_bot": True, "first_name": "Bot"},
                "text": "stub",
            },
        }
    )
    cb = cb.as_(bot)  # type: ignore[assignment]

    responses: list[str] = []

    async def message_answer(text=None, **kwargs):
        responses.append(text or "")
        return None

    acknowledgements: list[str | None] = []

    async def callback_answer(text=None, **kwargs):
        acknowledgements.append(text)
        return None

    object.__setattr__(cb.message, "answer", message_answer)
    object.__setattr__(cb, "answer", callback_answer)

    await main.process_request(cb, db, bot)  # type: ignore[arg-type]

    assert main.festival_edit_sessions[1] == (fid, main.FESTIVAL_EDIT_FIELD_ACTIVITIES)
    assert responses and "Предпросмотр" in responses[-1]
    assert acknowledgements == [None]

    main.festival_edit_sessions.pop(1, None)


def test_parse_invalid_yaml():
    with pytest.raises(FestivalActivitiesError):
        parse_festival_activities_yaml("version: 3")

