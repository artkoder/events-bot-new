import pytest

from db import Database
from models import Event, EventPoster
from smart_event_update import PosterCandidate, _apply_posters
from sqlalchemy import select


@pytest.mark.asyncio
async def test_apply_posters_prunes_foreign_posters_by_scope_hashes(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite")
    db = Database(db_path)
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Мёртвые души",
            description="desc",
            date="2026-02-07",
            time="18:00",
            location_name="Драматический театр",
            city="Калининград",
            source_text="src",
            photo_urls=[
                "https://files.catbox.moe/a.jpg",
                "https://files.catbox.moe/b.jpg",
                "https://files.catbox.moe/c.jpg",
            ],
            photo_count=3,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

        # Pretend a schedule/album post was incorrectly attached earlier: all posters got
        # added to this event.
        for h in ("a", "b", "c"):
            session.add(
                EventPoster(
                    event_id=int(ev.id),
                    poster_hash=h,
                    catbox_url=f"https://files.catbox.moe/{h}.jpg",
                )
            )
        await session.commit()

        # Now a new run decides only poster "b" belongs to this event.
        posters = [PosterCandidate(catbox_url="https://files.catbox.moe/b.jpg", sha256="b")]
        added, _added_urls, _preview_invalidated, pruned, _changed = await _apply_posters(
            session,
            int(ev.id),
            posters,
            poster_scope_hashes=["a", "b", "c"],
        )
        await session.commit()

        assert added == 0  # already existed
        assert pruned == 2

        refreshed = await session.get(Event, int(ev.id))
        assert refreshed is not None
        assert refreshed.photo_urls == ["https://files.catbox.moe/b.jpg"]
        assert refreshed.photo_count == 1

        res = await session.execute(
            select(EventPoster).where(EventPoster.event_id == int(ev.id))
        )
        remaining = res.scalars().all()
        assert len(remaining) == 1
        assert remaining[0].poster_hash == "b"
