import pytest

from db import Database
from models import Event, EventPoster
from smart_event_update import _apply_posters


@pytest.mark.asyncio
async def test_apply_posters_does_not_prune_scope_when_selection_empty(tmp_path) -> None:
    """
    Regression guard: if poster matching fails and `posters=[]` but `poster_scope_hashes`
    is provided, we must not delete existing posters. Empty selection usually means OCR/title
    matching failed, not that the user wants to remove images.
    """
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Event",
            description="desc",
            date="2026-02-07",
            time="18:00",
            location_name="Loc",
            city="Калининград",
            source_text="src",
            photo_urls=["https://files.catbox.moe/a.jpg"],
            photo_count=1,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

        session.add(
            EventPoster(
                event_id=int(ev.id),
                poster_hash="a",
                catbox_url="https://files.catbox.moe/a.jpg",
            )
        )
        await session.commit()

        added, _urls, _invalidated, pruned, _changed = await _apply_posters(
            session,
            int(ev.id),
            posters=[],  # selection empty
            poster_scope_hashes=["a"],
        )
        await session.commit()

        assert added == 0
        assert pruned == 0

        refreshed = await session.get(Event, int(ev.id))
        assert refreshed is not None
        assert refreshed.photo_count == 1
        assert refreshed.photo_urls == ["https://files.catbox.moe/a.jpg"]
