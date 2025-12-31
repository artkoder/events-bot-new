"""
Tests for special_pages module.

Coverage:
- Title normalization
- Event grouping/deduplication
- Ticket line formatting
- Location formatting
"""
import pytest
from datetime import date
from dataclasses import dataclass
from typing import Optional


# Mock Event class for testing (simplified)
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
    
    def __post_init__(self):
        if self.photo_urls is None:
            self.photo_urls = []


class TestNormalizeTitle:
    """Tests for normalize_title function."""
    
    def test_simple_title(self):
        from special_pages import normalize_title
        assert normalize_title("Concert") == "concert"
    
    def test_with_emoji(self):
        from special_pages import normalize_title
        assert normalize_title("🎭 Спектакль Гамлет") == "спектакль гамлет"
    
    def test_with_multiple_emoji(self):
        from special_pages import normalize_title
        assert normalize_title("🎵🎶 Концерт") == "концерт"
    
    def test_whitespace_collapse(self):
        from special_pages import normalize_title
        assert normalize_title("  Concert   with   spaces  ") == "concert with spaces"
    
    def test_case_insensitive(self):
        from special_pages import normalize_title
        assert normalize_title("CONCERT") == normalize_title("concert")
        assert normalize_title("Concert") == normalize_title("CONCERT")


class TestFormatTicketLine:
    """Tests for format_ticket_line function."""
    
    def test_free_event(self):
        from special_pages import format_ticket_line
        event = MockEvent(is_free=True)
        assert format_ticket_line(event) == "бесплатно"
    
    def test_free_with_registration(self):
        from special_pages import format_ticket_line
        event = MockEvent(is_free=True, ticket_link="https://example.com")
        assert format_ticket_line(event) == "бесплатно, регистрация"
    
    def test_sold_out(self):
        from special_pages import format_ticket_line
        event = MockEvent(ticket_status="sold_out")
        assert format_ticket_line(event) == "билеты проданы"
    
    def test_single_price(self):
        from special_pages import format_ticket_line
        event = MockEvent(ticket_price_min=500)
        assert format_ticket_line(event) == "500₽"
    
    def test_price_range(self):
        from special_pages import format_ticket_line
        event = MockEvent(ticket_price_min=300, ticket_price_max=1000)
        assert format_ticket_line(event) == "от 300₽"
    
    def test_registration_only(self):
        from special_pages import format_ticket_line
        event = MockEvent(ticket_link="https://example.com")
        assert format_ticket_line(event) == "регистрация"
    
    def test_no_ticket_info(self):
        from special_pages import format_ticket_line
        event = MockEvent()
        assert format_ticket_line(event) == ""


class TestFormatLocation:
    """Tests for format_location function."""
    
    def test_venue_only(self):
        from special_pages import format_location
        event = MockEvent(location_name="Drama Theatre")
        assert format_location(event) == "Drama Theatre"
    
    def test_venue_with_address(self):
        from special_pages import format_location
        event = MockEvent(location_name="Drama Theatre", location_address="ул. Ленина, 5")
        assert format_location(event) == "Drama Theatre, ул. Ленина, 5"
    
    def test_venue_with_city(self):
        from special_pages import format_location
        event = MockEvent(location_name="Drama Theatre", city="Калининград")
        assert format_location(event) == "Drama Theatre, Калининград"
    
    def test_full_location(self):
        from special_pages import format_location
        event = MockEvent(
            location_name="Drama Theatre",
            location_address="ул. Ленина, 5",
            city="Калининград"
        )
        assert format_location(event) == "Drama Theatre, ул. Ленина, 5, Калининград"


class TestGroupEventsForSpecial:
    """Tests for group_events_for_special function (deduplication)."""
    
    def test_dedup_same_title_different_times(self):
        """Two events with same title and different times should merge."""
        from special_pages import group_events_for_special
        
        event1 = MockEvent(id=1, title="Спектакль Гамлет", date="2026-01-01", time="14:00")
        event2 = MockEvent(id=2, title="Спектакль Гамлет", date="2026-01-01", time="19:00")
        
        # Need to mock parse_iso_date
        import sys
        import types
        
        # Create mock main module if not exists
        if 'main' not in sys.modules:
            mock_main = types.ModuleType('main')
            mock_main.parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
            sys.modules['main'] = mock_main
        
        grouped = group_events_for_special([event1, event2])
        
        assert len(grouped) == 1  # one day
        day = date(2026, 1, 1)
        assert day in grouped
        assert len(grouped[day]) == 1  # one group
        group = grouped[day][0]
        assert len(group.slots) == 2  # two time slots
        assert group.slots[0].time == "14:00"
        assert group.slots[1].time == "19:00"
    
    def test_different_titles_not_merged(self):
        """Events with different titles should not merge."""
        from special_pages import group_events_for_special
        
        event1 = MockEvent(id=1, title="Концерт", date="2026-01-01", time="18:00")
        event2 = MockEvent(id=2, title="Спектакль", date="2026-01-01", time="19:00")
        
        import sys
        import types
        if 'main' not in sys.modules:
            mock_main = types.ModuleType('main')
            mock_main.parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
            sys.modules['main'] = mock_main
        
        grouped = group_events_for_special([event1, event2])
        
        day = date(2026, 1, 1)
        assert len(grouped[day]) == 2  # two separate groups
    
    def test_same_title_different_days_not_merged(self):
        """Same title on different days should not merge."""
        from special_pages import group_events_for_special
        
        event1 = MockEvent(id=1, title="Концерт", date="2026-01-01", time="18:00")
        event2 = MockEvent(id=2, title="Концерт", date="2026-01-02", time="18:00")
        
        import sys
        import types
        if 'main' not in sys.modules:
            mock_main = types.ModuleType('main')
            mock_main.parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
            sys.modules['main'] = mock_main
        
        grouped = group_events_for_special([event1, event2])
        
        assert len(grouped) == 2  # two days
        assert date(2026, 1, 1) in grouped
        assert date(2026, 1, 2) in grouped
    
    def test_emoji_normalization_in_grouping(self):
        """Events with same title but different emoji should merge."""
        from special_pages import group_events_for_special
        
        event1 = MockEvent(id=1, title="🎭 Спектакль", date="2026-01-01", time="14:00")
        event2 = MockEvent(id=2, title="Спектакль", date="2026-01-01", time="19:00")
        
        import sys
        import types
        if 'main' not in sys.modules:
            mock_main = types.ModuleType('main')
            mock_main.parse_iso_date = lambda x: date.fromisoformat(x.split("..")[0])
            sys.modules['main'] = mock_main
        
        grouped = group_events_for_special([event1, event2])
        
        day = date(2026, 1, 1)
        assert len(grouped[day]) == 1  # merged into one group
        assert len(grouped[day][0].slots) == 2


class TestNormalizeTitleEmoji:
    """Test that emoji is properly removed during normalization."""
    
    def test_various_emoji(self):
        from special_pages import normalize_title
        
        assert normalize_title("🎵 Концерт") == "концерт"
        assert normalize_title("🎭 Спектакль") == "спектакль"
        assert normalize_title("🎪 Цирк 🎠") == "цирк"
        assert normalize_title("✨ Праздник ✨") == "праздник"
