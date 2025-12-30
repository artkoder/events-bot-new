
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from main import sync_month_page, Database
from models import MonthPage, Event

@pytest.mark.asyncio
async def test_sync_month_page_creates_new_page_with_update_links_true(tmp_path):
    # Setup DB
    db_path = tmp_path / "test_db.sqlite"
    db = Database(path=str(db_path))
    await db.init()
    
    # Pre-seed a "corrupted" or initialized-but-empty month page
    # This simulates the state where a deferred task runs for a new month
    month = "2026-05"
    async with db.get_session() as session:
        page = MonthPage(month=month, url="", path="", content_hash="")
        session.add(page)
        # Add an event so there is content to generate
        event = Event(
            title="Concert", 
            description="Test", 
            date=f"{month}-15", 
            time="19:00", 
            location_name="Club",
            source_text="Test source"
        )
        session.add(event)
        await session.commit()

    # Mock Telegraph
    # We need to patch 'main_part2.Telegraph' or wherever it is used.
    # Since main_part2 is exec'd into main, we might need to patch 'main.telegraph_create_page' directly 
    # if it intercepts the call, OR patch 'main.Telegraph'.
    # The function calls 'telegraph_create_page' which is in main.
    
    with patch("main.telegraph_create_page", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {"url": "https://telegra.ph/Test-May-2026", "path": "Test-May-2026"}
        
        # We also need to patch 'telegraph_edit_page' to ensure it's NOT called?
        # Or just verify create IS called.
        
        # ACT: Call sync_month_page with update_links=True
        # This was the condition that caused the bug (early return)
        result = await sync_month_page(db, month, update_links=True, force=True)
        
        # ASSERT
        assert result is True, "sync_month_page should return True (changes made)"
        
        # Verify create was called
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        assert "Concert" in kwargs["html_content"]
        
        # Verify DB updated
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
            assert page.url == "https://telegra.ph/Test-May-2026"
            assert page.path == "Test-May-2026"

@pytest.mark.asyncio
async def test_sync_month_page_skips_update_links_if_path_empty(tmp_path):
    """Specific test for the fix: ensuring we don't return early in update_links block."""
    db_path = tmp_path / "test_db_2.sqlite"
    db = Database(path=str(db_path))
    await db.init()
    
    month = "2026-06"
    async with db.get_session() as session:
        # Empty path
        page = MonthPage(month=month, url="", path="", content_hash="")
        session.add(page)
        session.add(Event(title="Fest", description="Desc", date=f"{month}-01", time="10:00", location_name="Loc", source_text="Src"))
        await session.commit()
        
    with patch("main.telegraph_create_page", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = {"url": "https://t.ph/new", "path": "new"}
        
        # We also mock build_month_nav_block to verify it is NOT called if we skip the block
        with patch("main.build_month_nav_block", new_callable=AsyncMock) as mock_nav:
            
            await sync_month_page(db, month, update_links=True, force=True)
            
            # Since page.path was empty, the code should SKIP the update_links block
            # So build_month_nav_block should NOT be called?
            # Wait, my fix was 'if update_links and page.path:'
            # So 'build_month_nav_block' (line 544) is inside the block.
            # So it should NOT be called.
            
            mock_nav.assert_not_called()
            mock_create.assert_called_once()
