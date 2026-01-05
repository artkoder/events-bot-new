"""Tests for 3D preview display on festival page."""
from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from db import Database
from models import Festival, Event


def _make_event(
    id: int,
    title: str,
    festival: str,
    event_date: str,
    preview_3d_url: str | None = None,
    photo_urls: list[str] | None = None,
) -> Event:
    return Event(
        id=id,
        title=title,
        description="Test event description",
        location_name="Test Location",
        source_text="Test source",
        festival=festival,
        date=event_date,
        time="19:00",
        photo_urls=photo_urls or [],
        preview_3d_url=preview_3d_url,
    )


@pytest.mark.asyncio
async def test_build_festival_page_content_shows_3d_preview(tmp_path: Path, monkeypatch):
    """Event with preview_3d_url should display figure with 3D preview above title."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    
    async with db.get_session() as session:
        fest = Festival(name="TestFest", full_name="Test Festival", start_date=tomorrow, end_date=tomorrow)
        session.add(fest)
        await session.commit()
        
        ev_with_preview = _make_event(
            id=1,
            title="Event With Preview",
            festival="TestFest",
            event_date=tomorrow,
            preview_3d_url="https://example.com/preview3d.jpg",
            photo_urls=["https://example.com/photo.jpg"],
        )
        session.add(ev_with_preview)
        await session.commit()
    
    # Patch generate_festival_description to avoid LLM calls
    async def mock_generate_desc(*a, **k):
        return None
    monkeypatch.setattr(main, "generate_festival_description", mock_generate_desc)
    
    async with db.get_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(Festival).where(Festival.name == "TestFest"))
        fest = result.scalar_one()
        
        title, nodes = await main.build_festival_page_content(db, fest)
    
    # Find figure node with 3D preview URL
    figure_nodes = [n for n in nodes if n.get("tag") == "figure"]
    assert len(figure_nodes) >= 1, "Should have at least one figure node"
    
    # Check if any figure contains the 3D preview URL
    preview_found = False
    for fig in figure_nodes:
        children = fig.get("children", [])
        for child in children:
            if child.get("tag") == "img":
                src = child.get("attrs", {}).get("src", "")
                if src == "https://example.com/preview3d.jpg":
                    preview_found = True
                    break
    
    assert preview_found, "3D preview image should be displayed in a figure node"


@pytest.mark.asyncio
async def test_build_festival_page_content_no_image_without_preview(tmp_path: Path, monkeypatch):
    """Event without preview_3d_url should not display figure for event image."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    
    async with db.get_session() as session:
        fest = Festival(name="TestFest2", full_name="Test Festival 2", start_date=tomorrow, end_date=tomorrow)
        session.add(fest)
        await session.commit()
        
        ev_without_preview = _make_event(
            id=2,
            title="Event Without Preview",
            festival="TestFest2",
            event_date=tomorrow,
            preview_3d_url=None,
            photo_urls=["https://example.com/photo.jpg"],
        )
        session.add(ev_without_preview)
        await session.commit()
    
    async def mock_generate_desc(*a, **k):
        return None
    monkeypatch.setattr(main, "generate_festival_description", mock_generate_desc)
    
    async with db.get_session() as session:
        from sqlalchemy import select
        result = await session.execute(select(Festival).where(Festival.name == "TestFest2"))
        fest = result.scalar_one()
        
        title, nodes = await main.build_festival_page_content(db, fest)
    
    # Find figure nodes with event photo URL (not festival cover)
    event_photo_figures = []
    for n in nodes:
        if n.get("tag") == "figure":
            children = n.get("children", [])
            for child in children:
                if child.get("tag") == "img":
                    src = child.get("attrs", {}).get("src", "")
                    if src == "https://example.com/photo.jpg":
                        event_photo_figures.append(n)
    
    # Should not display event photo as figure when there's no 3D preview
    assert len(event_photo_figures) == 0, "Event photo should not be displayed without 3D preview"
