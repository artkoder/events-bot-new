import pytest
from unittest.mock import MagicMock, AsyncMock
from main import optimize_month_chunks
from models import Event

@pytest.mark.asyncio
async def test_optimize_month_chunks_atomic_date():
    """
    Test that splitting respects date boundaries.
    Scenario:
    - Day 1: 10 events (Size ~1000)
    - Day 2: 10 events (Size ~1000)
    - Limit: ~1500 (enough for 1.5 days)
    
    Naive split: Would split at 15 events (Day 2, Event 5).
    Atomic split: Must split at 10 events (End of Day 1).
    """
    
    # 1. Setup Mock DB & Nav
    mock_db = MagicMock()
    mock_nav = "<div>Nav</div>"
    
    # 2. Setup Events
    events = []
    # Day 1: 10 events
    for i in range(10):
        e = Event(
            date="2026-01-01",
            time=f"10:{i:02d}",
            title=f"Event Day 1 #{i}",
            slug=f"evt-1-{i}"
        )
        events.append(e)
        
    # Day 2: 10 events
    for i in range(10):
        e = Event(
            date="2026-01-02", 
            time=f"10:{i:02d}",
            title=f"Event Day 2 #{i}",
            slug=f"evt-2-{i}"
        )
        events.append(e)
        
    exhibitions = []
    
    # 3. Mock main.py dependencies
    # We need to ensure build_month_page_content returns a size proportional to events
    # Real build_month_page_content creates nodes.
    # To avoid complex mocking of build_month_page_content interactions (which calls DB),
    # we might need to rely on the fact that existing tests import main.
    # PROBABLE ISSUE: optimize_month_chunks calls build_month_page_content which queries DB for some things?
    # No, it passes events. But build_month_page_content might query 'MonthPage' or others?
    # Let's check build_month_page_content signature in main.py.
    # It takes (db, month, events, ...).
    # Inside it might do extra things. 
    # But wait! optimizing_month_chunks calls `build_month_page_content`.
    # If we want to test the logic WITHOUT depending on real rendering size, we can mock `build_month_page_content`?
    # But `optimize_month_chunks` calculates size based on the RESULT of `build_month_page_content`.
    # So we should mock `build_month_page_content` to return a specific size based on input length.
    
    import main
    
    # We will mock the function `build_month_page_content` in `main` module scope
    original_build = main.build_month_page_content
    
    async def mock_build(db, month, evts, exhs, **kwargs):
        # Return dummy content
        # We simulate that each event adds 100 bytes.
        # Overhead + Nav = 500 bytes.
        size = 500 + (len(evts) * 100)
        
        # We need to return (title, content, ...)
        # content must be a list of nodes.
        # nodes_to_html needs to work on it.
        # We can construct a dummy node covering the size.
        dummy_node = {'tag': 'p', 'children': ['a' * size]}
        return "Title", [dummy_node], None

    main.build_month_page_content = mock_build
    
    # Force TELEGRAPH_LIMIT to fit 15 events (500 + 1500 = 2000)
    # 10 events = 1500 bytes.
    # 20 events = 2500 bytes.
    # We set limit to 2000.
    # So 15 events fits (2000). 16 events (2100) fails.
    # Naive binary search will try to fit ~15.
    # Atomic logic should rollback to 10 (Day 1 end) because 15 includes half of Day 2.
    
    original_limit = main.TELEGRAPH_LIMIT
    main.TELEGRAPH_LIMIT = 2000
    
    try:
        chunks, _, _ = await optimize_month_chunks(mock_db, "2026-01", events, exhibitions, mock_nav)
        
        print(f"Chunks count: {len(chunks)}")
        for i, (chnk_evts, _) in enumerate(chunks):
            print(f"Chunk {i} events: {len(chnk_evts)}")
            if chnk_evts:
                print(f"  First: {chnk_evts[0].date}, Last: {chnk_evts[-1].date}")
        
        # ASSERTIONS
        # We expect Chunk 1 to have exactly 10 events (all of Day 1).
        # Chunk 2 to have 10 events (all of Day 2).
        
        # If Logic is Naive: Chunk 1 will have ~15 events (Day 1 + half Day 2).
        
        first_chunk_events = chunks[0][0]
        assert len(first_chunk_events) == 10, f"Expected 10 events in first chunk, got {len(first_chunk_events)}"
        assert first_chunk_events[-1].date == "2026-01-01"
        
        # Verify second chunk starts with Day 2
        second_chunk_events = chunks[1][0]
        assert second_chunk_events[0].date == "2026-01-02"
        
    finally:
        # Restore
        main.build_month_page_content = original_build
        main.TELEGRAPH_LIMIT = original_limit

