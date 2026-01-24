import pytest
from aiogram import types
from db import Database
from main_part2 import event_to_nodes
from models import Event, User
from preview_3d.handlers import handle_3di_command, update_previews_from_results


DENIED_TEXT = "\u274c \u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043f\u0440\u0430\u0432"


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": kwargs.get("reply_markup"),
            }
        )


def _make_event(event_id: int, **overrides: object) -> Event:
    payload = {
        "id": event_id,
        "title": f"Event {event_id}",
        "description": "Desc",
        "date": "2026-05-15",
        "time": "19:00",
        "location_name": "Loc",
        "source_text": "src",
    }
    payload.update(overrides)
    return Event(**payload)


def test_event_to_nodes_prefers_preview_3d_url():
    event = _make_event(
        1,
        photo_urls=["https://example.com/photo.jpg"],
        preview_3d_url="https://example.com/preview.jpg",
    )
    nodes = event_to_nodes(event, show_image=True)
    assert nodes[0]["tag"] == "figure"
    assert nodes[0]["children"][0]["attrs"]["src"] == "https://example.com/preview.jpg"


def test_event_to_nodes_falls_back_to_photo_urls():
    event = _make_event(
        1,
        photo_urls=["https://example.com/photo.jpg"],
        preview_3d_url="not-a-url",
    )
    nodes = event_to_nodes(event, show_image=True)
    assert nodes[0]["tag"] == "figure"
    assert nodes[0]["children"][0]["attrs"]["src"] == "https://example.com/photo.jpg"


@pytest.mark.asyncio
async def test_3di_command_denied_without_superadmin(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=False))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "/3di",
        }
    )
    bot = DummyBot()
    await handle_3di_command(msg, db, bot)

    assert bot.messages
    assert bot.messages[0]["text"] == DENIED_TEXT


@pytest.mark.asyncio
async def test_update_previews_from_results_updates_db(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        session.add(_make_event(2))
        await session.commit()

    results = [
        {
            "event_id": 1,
            "preview_url": "https://example.com/preview.jpg",
            "status": "ok",
        },
        {
            "event_id": 2,
            "preview_url": "https://example.com/other.jpg",
            "status": "error",
            "error": "render failed",
        },
    ]
    updated, errors, skipped = await update_previews_from_results(db, results)

    assert updated == 1
    assert errors == 1
    assert skipped == 0

    async with db.get_session() as session:
        ev1 = await session.get(Event, 1)
        ev2 = await session.get(Event, 2)

    assert ev1.preview_3d_url == "https://example.com/preview.jpg"
    assert ev2.preview_3d_url is None


@pytest.mark.asyncio
async def test_update_previews_from_results_handles_skip(tmp_path):
    """Test that skip status is counted separately from errors."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        await session.commit()

    results = [
        {"event_id": 1, "status": "skip", "error": "No images"},
    ]
    updated, errors, skipped = await update_previews_from_results(db, results)

    assert updated == 0
    assert errors == 0
    assert skipped == 1


@pytest.mark.asyncio
async def test_get_new_events_gap(tmp_path):
    """Test that _get_new_events_gap returns events after the last one with preview."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    from preview_3d.handlers import _get_new_events_gap

    # Create 4 events:
    # 4 (Newest) - No preview, 2 images -> Should return
    # 3          - No preview, 0 images -> Should skip (if min_images=1)
    # 2          - Has preview          -> Stop barrier
    # 1 (Oldest) - No preview           -> Should not reach
    
    async with db.get_session() as session:
        session.add(_make_event(1, photo_urls=["http://img"], preview_3d_url=None))
        session.add(_make_event(2, photo_urls=["http://img"], preview_3d_url="http://preview"))
        session.add(_make_event(3, photo_urls=[], preview_3d_url=None))
        session.add(_make_event(4, photo_urls=["http://img"], preview_3d_url=None))
        await session.commit()
        
    candidates = await _get_new_events_gap(db, min_images=1)
    
    # Updated logic: does NOT stop at barrier.
    # Should get 4 (newest) AND 1 (oldest, behind barrier w/ proper images).
    # Event 3 skipped due to no images.
    
    assert len(candidates) == 2
    ids = sorted([e.id for e in candidates])
    assert ids == [1, 4]
    
    # Test with min_images=0, should get 4, 3, 1
    candidates_all = await _get_new_events_gap(db, min_images=0)
    assert len(candidates_all) == 3
    ids = sorted([e.id for e in candidates_all])
    assert ids == [1, 3, 4]

