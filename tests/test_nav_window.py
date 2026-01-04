
import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from main import build_month_nav_html, Database, Event, MonthPage

@pytest.mark.asyncio
async def test_nav_window_includes_april():
    # Mock database and session
    mock_db = MagicMock(spec=Database)
    mock_session = MagicMock()
    mock_db.get_session.return_value.__aenter__.return_value = mock_session
    
    # Mock date.today() to be December 30, 2025
    with patch('main.datetime') as mock_datetime, \
         patch('main.date') as mock_date:
        
        # Setup fixed today
        fixed_today = date(2025, 12, 30)
        mock_datetime.now.return_value.date.return_value = fixed_today
        mock_date.today.return_value = fixed_today
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

        # We need to spy on the SQL query to see what date range is used
        # or we can mock the results to see if they are filtered.
        # Ideally, we check the query parameters, but querying actual sqlite is better for integration.
        # Let's try to infer from the SQL generated or just check result processing.
        
        # However, `build_month_nav_html` constructs a query. 
        # Let's check the `end_nav` variable logic by running the function and seeing if it asks for April data.
        
        # Using a valid DB is safer. Let's use an in-memory DB or the `dbsession` fixture if available.
        # Since I don't see conftest locally easily, I will write a standalone-ish test 
        # that relies on `main` importing `Event` and `MonthPage`.
        
        pass

# Better approach: modify main_part2 reference or main.py logic?
# The logic is in main.py: build_month_nav_html
# Let's write a test that uses real aiosqlite in memory to be sure.

import aiosqlite
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Define a local Base for the test if models.py doesn't export one cleanly,
# or better yet, verify models.py content.
# Based on common patterns, models usually inherit from a DeclarativeBase.
# Let's inspect models.py first (via view_file) but to fix blindly:

from sqlmodel import SQLModel

@pytest.mark.asyncio
async def test_nav_window_logic():
    # Setup in-memory DB
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Insert events: January 2026 and April 2026
    async with Session() as session:
        session.add(Event(
            title="Jan Event", 
            date="2026-01-15", 
            source_text="s", 
            location_name="l",
            description="d",
            time="19:00"
        ))
        session.add(Event(
            title="Apr Event", 
            date="2026-04-18", 
            source_text="s", 
            location_name="l",
            description="d",
            time="19:00"
        ))
        session.add(MonthPage(month="2026-01", url="http://jan", path="jan"))
        session.add(MonthPage(month="2026-04", url="http://apr", path="apr"))
        await session.commit()
    
    # Mock DB wrapper
    mock_db = MagicMock(spec=Database)
    mock_db.get_session = MagicMock(side_effect=lambda: Session())
    
    # Mock time to Dec 2025
    with patch('main.datetime') as mock_datetime, \
         patch('main.date', wraps=date) as mock_date:
        
        fixed_today = date(2025, 12, 30)
        mock_datetime.now.return_value.date.return_value = fixed_today
        # main.py uses logic: end_nav = date(today.year + 1, 4, 1) -> 2026-04-01
        
        # Run function
        html_out = await build_month_nav_html(mock_db)
        
        # Expectation: 
        # If buggy: April link ("2026-04") is NOT in html_out
        # If fixed: April link IS in html_out
        
        print(f"DEBUG: html_out = {html_out}")
        
        # Assertion: We WANT April to be there.
        # But for reproduction, we expect this to FAIL if the code isn't fixed yet.
        # So we assert it exists, and expect pytest to fail.
        assert "href=\"http://apr\"" in html_out, "April link missing from navigation!"
        assert "апрель" in html_out, "April text missing from navigation!"
        
        return html_out

