import pytest

from db import Database
from models import Event
from smart_event_update import PosterCandidate, _apply_posters


@pytest.mark.asyncio
async def test_apply_posters_prefers_ocr_title_match_over_ocr_length(tmp_path) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    url_unrelated = "https://files.catbox.moe/unrelated.jpg"
    url_relevant = "https://files.catbox.moe/relevant.jpg"
    url_no_ocr = "https://files.catbox.moe/no-ocr.jpg"

    async with db.get_session() as session:
        ev = Event(
            title="Хорошо бы после этого проснуться",
            description="desc",
            date="2026-02-24",
            time="19:00",
            location_name="Loc",
            city="Калининград",
            source_text="src",
            photo_urls=[url_unrelated, url_relevant, url_no_ocr],
            photo_count=3,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

        posters = [
            PosterCandidate(
                catbox_url=url_unrelated,
                sha256="a",
                ocr_text=(
                    "Большой концерт и фестиваль, скидка 50% по промокоду.\n"
                    "Очень много текста, но это НЕ про это событие.\n"
                    "Ещё строка, чтобы OCR было длиннее."
                ),
            ),
            PosterCandidate(
                catbox_url=url_relevant,
                sha256="b",
                ocr_text="Хорошо бы после этого проснуться\n24/02 19:00",
            ),
            PosterCandidate(
                catbox_url=url_no_ocr,
                sha256="c",
            ),
        ]

        _added, _urls, _invalidated, _pruned, _changed = await _apply_posters(
            session,
            int(ev.id),
            posters,
        )
        await session.commit()

    async with db.get_session() as session:
        refreshed = await session.get(Event, int(ev.id))
        assert refreshed is not None
        assert refreshed.photo_urls
        assert refreshed.photo_urls[0] == url_relevant
        assert set(refreshed.photo_urls) == {url_unrelated, url_relevant, url_no_ocr}
