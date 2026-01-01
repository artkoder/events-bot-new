import pytest
from datetime import date
from dataclasses import dataclass
from typing import Optional

# Mock Event class relevant for this test
@dataclass
class MockEvent:
    id: int = 1
    title: str = "Test Event"
    description: str = "Test description"
    search_digest: Optional[str] = None
    date: str = "2026-01-01"
    time: str = "18:00"
    location_name: str = "Test Venue"
    location_address: Optional[str] = None
    city: Optional[str] = None
    ticket_price_min: Optional[int] = None
    ticket_price_max: Optional[int] = None
    ticket_link: Optional[str] = None
    ticket_status: Optional[str] = None
    is_free: bool = False
    telegraph_url: Optional[str] = None
    telegraph_path: Optional[str] = None
    source_post_url: Optional[str] = None
    ics_url: Optional[str] = None
    ics_post_url: Optional[str] = None
    photo_urls: list = None
    festival: Optional[str] = None
    end_date: Optional[str] = None
    pushkin_card: bool = False
    preview_3d_url: Optional[str] = None # Support for 3d preview
    
    def __post_init__(self):
        if self.photo_urls is None:
            self.photo_urls = []
    
    def model_dump(self):
        return self.__dict__

def test_preview_3d_priority():
    from special_pages import group_events_for_special
    
    # Event with both photo_urls and preview_3d_url
    event1 = MockEvent(
        id=1, 
        title="Event with 3D", 
        date="2026-01-01", 
        time="18:00",
        photo_urls=["http://example.com/photo1.jpg"],
        preview_3d_url="http://example.com/preview3d.jpg"
    )
    
    # Mock main module dependencies
    import sys
    import types
    if 'main' not in sys.modules:
        mock_main = types.ModuleType('main')
        mock_main.parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
        sys.modules['main'] = mock_main
    else:
        # ensure parse_iso_date exists
        if not hasattr(sys.modules['main'], 'parse_iso_date'):
            sys.modules['main'].parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
    
    grouped = group_events_for_special([event1])
    day = date(2026, 1, 1)
    
    # The group's photo_url should be the preview_3d_url
    assert grouped[day][0].photo_url == "http://example.com/preview3d.jpg"

def test_preview_3d_only():
    from special_pages import group_events_for_special
    
    # Event with only preview_3d_url
    event1 = MockEvent(
        id=1, 
        title="Event with 3D only", 
        date="2026-01-01", 
        time="18:00",
        photo_urls=[],
        preview_3d_url="http://example.com/preview3d_only.jpg"
    )
    
    import sys
    if not hasattr(sys.modules['main'], 'parse_iso_date'):
        sys.modules['main'].parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])

    grouped = group_events_for_special([event1])
    day = date(2026, 1, 1)
    
    assert grouped[day][0].photo_url == "http://example.com/preview3d_only.jpg"

def test_no_preview_fallback():
    from special_pages import group_events_for_special
    
    # Event with only photo_urls
    event1 = MockEvent(
        id=1, 
        title="Event with photo only", 
        date="2026-01-01", 
        time="18:00",
        photo_urls=["http://example.com/photo1.jpg"],
        preview_3d_url=None
    )
    
    import sys
    if not hasattr(sys.modules['main'], 'parse_iso_date'):
        sys.modules['main'].parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])

    grouped = group_events_for_special([event1])
    day = date(2026, 1, 1)
    
    assert grouped[day][0].photo_url == "http://example.com/photo1.jpg"
