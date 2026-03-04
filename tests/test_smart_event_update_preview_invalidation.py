import pytest

from db import Database
from models import Event
from smart_event_update import PosterCandidate, _apply_posters


def _make_event(event_id: int, **overrides: object) -> Event:
    payload = {
        "id": event_id,
        "title": "Event",
        "description": "Desc",
        "date": "2026-05-15",
        "time": "19:00",
        "location_name": "Loc",
        "source_text": "src",
        "photo_urls": ["https://example.com/a.jpg"],
        "photo_count": 1,
        "preview_3d_url": "https://example.com/preview.jpg",
    }
    payload.update(overrides)
    return Event(**payload)


@pytest.mark.asyncio
async def test_apply_posters_invalidates_preview_3d_when_images_change(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        await session.commit()

    async with db.get_session() as session:
        added, urls, invalidated, pruned, changed = await _apply_posters(
            session,
            1,
            [
                PosterCandidate(catbox_url="https://example.com/b.jpg"),
            ],
        )
        assert added == 0
        assert urls == []
        assert invalidated is True
        assert pruned == 0
        assert changed is True
        await session.commit()

    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev is not None
        assert ev.preview_3d_url is None
        assert "https://example.com/b.jpg" in (ev.photo_urls or [])
