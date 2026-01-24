import json
from datetime import datetime, timezone

import pytest

from db import Database
from models import Event, EventPoster
from video_announce import poster_overlay


@pytest.mark.asyncio
async def test_enrich_payload_with_poster_overlays_adds_overlay_and_drops_empty_ocr(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ok = Event(
            title="Cool Concert",
            description="d",
            source_text="s",
            date="2026-01-01",
            time="19:00",
            location_name="Big Hall",
            city="City",
            photo_urls=["https://example.com/1.jpg"],
            photo_count=1,
        )
        bad = Event(
            title="Bad Poster",
            description="d",
            source_text="s",
            date="2026-01-01",
            time="20:00",
            location_name="Somewhere",
            city="City",
            photo_urls=["https://example.com/2.jpg"],
            photo_count=1,
        )
        session.add(ok)
        session.add(bad)
        await session.commit()
        await session.refresh(ok)
        await session.refresh(bad)

        session.add(
            EventPoster(
                event_id=ok.id,
                poster_hash="h1",
                ocr_text="19:00",  # deliberately incomplete, but non-empty
                ocr_title="",
                updated_at=datetime.now(timezone.utc),
            )
        )
        session.add(
            EventPoster(
                event_id=bad.id,
                poster_hash="h2",
                ocr_text="...",  # punctuation-only -> treated as empty
                ocr_title="BAD",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    async def _fake_gemma_check(*, ocr_text: str, title: str, date: str, location: str):  # noqa: ARG001
        return poster_overlay.PosterCheck(
            has_title=False,
            has_date=False,
            has_time=True,
            has_location=False,
            missing=("title", "date", "location"),
            rationale="fake",
            model="fake",
        )

    monkeypatch.setattr(poster_overlay, "_gemma_check", _fake_gemma_check)

    payload = {
        "intro": {"text": "AFISHA", "count": 2},
        "scenes": [
            {
                "event_id": ok.id,
                "title": ok.title,
                "about": "About",
                "description": "",
                "date": "1 января 19:00",
                "location": "City, Big Hall",
                "images": ok.photo_urls,
                "is_free": False,
            },
            {
                "event_id": bad.id,
                "title": bad.title,
                "about": "About",
                "description": "",
                "date": "1 января 20:00",
                "location": "City, Somewhere",
                "images": bad.photo_urls,
                "is_free": False,
            },
        ],
    }

    out = await poster_overlay.enrich_payload_with_poster_overlays(
        db, json.dumps(payload, ensure_ascii=False)
    )
    out_obj = json.loads(out)
    assert out_obj["intro"]["count"] == 1
    assert len(out_obj["scenes"]) == 1
    scene = out_obj["scenes"][0]
    assert scene["event_id"] == ok.id
    assert "poster_overlay" in scene
    assert scene["poster_overlay"]["missing"] == ["title", "date", "location"]
    assert scene["poster_overlay"]["text"].splitlines() == [
        "Cool Concert",
        "1 января 19:00",
        "City, Big Hall",
    ]

