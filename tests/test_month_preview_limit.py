import pytest
from datetime import date
from unittest.mock import MagicMock
import main_part2
from main_part2 import _build_month_page_content_sync
from models import Event

def parse_iso_date(d_str):
    if not d_str: return None
    return date.fromisoformat(d_str)

# Inject dependency
main_part2.parse_iso_date = parse_iso_date
main_part2.MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
main_part2.MONTHS_GEN = ["", "января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
# Mock functions used
main_part2.ensure_event_telegraph_link = lambda e, f, db: None
main_part2.month_name_prepositional = lambda m: m
main_part2.LOCAL_TZ = None # or timezone.utc if needed
main_part2.today = lambda: date(2026, 5, 1) # mock today? No, it uses datetime.now(LOCAL_TZ).date()

# We need to mock datetime inside main_part2?
# main_part2 imports datetime.
from datetime import datetime, timezone
main_part2.LOCAL_TZ = timezone.utc
main_part2.format_day_pretty = lambda d: d.isoformat()
main_part2.telegraph_br = lambda: []
main_part2.PERM_START = {}
main_part2.PERM_END = {}
main_part2.rough_size = lambda x: 100
main_part2.DAY_START = lambda d: {}
main_part2.DAY_END = lambda d: {}
main_part2.is_recent = lambda e: False
main_part2.format_event_md = lambda *args, **kwargs: "MD CONTENT"
main_part2.md_to_html = lambda md: "HTML CONTENT"


def _create_dummy_events(count: int, start_id: int = 1) -> list[Event]:
    events = []
    for i in range(count):
        e = Event(
            id=start_id + i,
            title=f"Event {start_id + i}",
            date="2026-05-15",
            time="19:00",
            location_name="Loc",
            event_type="concert",
            photo_urls=["http://example.com/img.jpg"]
        )
        events.append(e)
    return events

def test_month_page_show_images_under_limit():
    """Test that validating the new limit (30): 15 events should SHOW images."""
    month = "2026-05"
    events = _create_dummy_events(15)
    exhibitions = []
    fest_map = {}
    
    # We call the sync function directly
    # It requires many arguments, we pass defaults/None where appropriate
    title, content, size = _build_month_page_content_sync(
        month=month,
        events=events,
        exhibitions=exhibitions,
        fest_map=fest_map,
        continuation_url=None,
        size_limit=None,
        fest_index_url=None,
        include_ics=True,
        include_details=True
    )
    
    # Check for 'figure' tag in content which indicates an image is rendered
    # We need to traverse the nodes.
    # The content is a list of nodes (dicts or strings).
    # We look for {'tag': 'figure', ...}
    
    has_images = False
    
    # Helper to search recursively if needed, but here structure is flat-ish usually
    # content is list of nodes.
    # event_to_nodes returns a list of nodes. 
    # If show_images=True, it adds a figure.
    
    def find_figure(nodes):
        for node in nodes:
            if isinstance(node, dict):
                if node.get("tag") == "figure":
                    return True
                if "children" in node:
                    if find_figure(node["children"]):
                        return True
        return False

    has_images = find_figure(content)
    
    # Expectation: 15 events < 30, so show_images should be True
    # CURRENTLY this will FAIL (limit is 10)
    assert has_images, "Images should be shown for 15 events (limit increased to 30)"

def test_month_page_hide_images_over_limit():
    """Test that validating the new limit (30): 35 events should HIDE images."""
    month = "2026-05"
    events = _create_dummy_events(35)
    exhibitions = []
    fest_map = {}
    
    title, content, size = _build_month_page_content_sync(
        month=month,
        events=events,
        exhibitions=exhibitions,
        fest_map=fest_map,
        continuation_url=None,
        size_limit=None,
        fest_index_url=None,
        include_ics=True,
        include_details=True
    )
    
    def find_figure(nodes):
        for node in nodes:
            if isinstance(node, dict):
                if node.get("tag") == "figure":
                    return True
                if "children" in node:
                    if find_figure(node["children"]):
                        return True
        return False

    has_images = find_figure(content)
    
    assert not has_images, "Images should be HIDDEN for 35 events (> 30)"
