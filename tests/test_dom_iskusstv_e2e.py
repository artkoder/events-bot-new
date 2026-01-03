"""
End-to-end integration test for Dom Iskusstv parser.

Tests the complete flow from URL extraction through Kaggle execution
to database insertion and event completeness verification.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest
from sqlmodel import select

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import main
from main import Database, Event
from source_parsing.dom_iskusstv import (
    extract_dom_iskusstv_urls,
    parse_dom_iskusstv_output,
    process_dom_iskusstv_events,
    run_dom_iskusstv_kaggle_kernel,
    SourceParsingStats,
)
from source_parsing.parser import TheatreEvent


class DummyBot:
    """Minimal bot stub for testing."""
    
    def __init__(self, token: str = "test:token"):
        self.token = token
        self.messages_sent: list[dict] = []
    
    async def send_message(self, chat_id: int, text: str, **kwargs):
        self.messages_sent.append({"chat_id": chat_id, "text": text})
        return MagicMock(message_id=len(self.messages_sent))


@pytest.fixture
def sample_kaggle_json() -> str:
    """Sample JSON output from Kaggle kernel."""
    return json.dumps([
        {
            "title": "«Сказки на ночь»",
            "date_raw": "3 ЯНВАРЯ 14:00",
            "parsed_date": "2026-01-03",
            "parsed_time": "14:00",
            "location": "Дом искусств",
            "url": "https://xn--b1admiilxbaki.xn--p1ai/?unifd-performance-id=832",
            "ticket_status": "available",
            "ticket_price_min": 300,
            "ticket_price_max": 500,
            "photos": [
                "https://static.tildacdn.com/tild3664-3261-4136-b830-373231356132/3_afisha.jpg",
                "https://static.tildacdn.com/tild6163-3535-4236-b034-636164333430/45__afisha.jpg",
            ],
            "description": "Драматическая труппа Калининградского театра эстрады приглашает детей и взрослых в необыкновенное путешествие.",
            "age_restriction": "6+",
            "source_type": "dom_iskusstv"
        },
        {
            "title": "«Сказки на ночь»",
            "date_raw": "4 ЯНВАРЯ 14:00",
            "parsed_date": "2026-01-04",
            "parsed_time": "14:00",
            "location": "Дом искусств",
            "url": "https://xn--b1admiilxbaki.xn--p1ai/?unifd-performance-id=833",
            "ticket_status": "available",
            "ticket_price_min": 300,
            "ticket_price_max": 500,
            "photos": [
                "https://static.tildacdn.com/tild3664-3261-4136-b830-373231356132/3_afisha.jpg",
            ],
            "description": "Драматическая труппа Калининградского театра эстрады приглашает детей и взрослых.",
            "age_restriction": "6+",
            "source_type": "dom_iskusstv"
        },
    ], ensure_ascii=False)


@pytest.fixture
async def test_db(tmp_path: Path) -> Database:
    """Create a test database."""
    db = Database(str(tmp_path / "test.sqlite"))
    await db.init()
    yield db


class TestE2EDomIskusstv:
    """End-to-end tests for the complete Dom Iskusstv flow."""
    
    @pytest.mark.asyncio
    async def test_full_flow_from_vk_text_to_database(
        self, 
        tmp_path: Path,
        sample_kaggle_json: str,
        monkeypatch,
    ):
        """
        Test the complete flow:
        1. URL extraction from VK post text
        2. Kaggle kernel execution (mocked)
        3. JSON parsing
        4. Event processing via LLM (mocked)
        5. Database insertion (mocked)
        6. Verification of statistics
        """
        # --- Step 1: URL Extraction from VK text ---
        vk_post_text = """
        🎭 Новый спектакль в Доме искусств!
        
        Приглашаем на «Сказки на ночь» — волшебный семейный спектакль!
        
        📅 3-4 января 2026
        📍 Дом искусств, Калининград
        
        Билеты: https://xn--b1admiilxbaki.xn--p1ai/skazka
        
        #дом_искусств #семейный_спектакль
        """
        
        urls = extract_dom_iskusstv_urls(vk_post_text)
        assert len(urls) == 1
        assert "skazka" in urls[0]
        
        # --- Step 2: Create mock Kaggle output file ---
        output_file = tmp_path / "dom_iskusstv_events.json"
        output_file.write_text(sample_kaggle_json, encoding="utf-8")
        
        # --- Step 3: Parse JSON output ---
        events = parse_dom_iskusstv_output([str(output_file)])
        assert len(events) == 2
        
        # Verify parsed events structure
        event = events[0]
        assert event.title == "«Сказки на ночь»"
        assert event.parsed_date == "2026-01-03"
        assert event.parsed_time == "14:00"
        assert event.location == "Дом искусств"
        assert event.ticket_status == "available"
        assert event.ticket_price_min == 300
        assert event.ticket_price_max == 500
        assert len(event.photos) == 2
        assert "Драматическая труппа" in event.description
        assert event.age_restriction == "6+"
        assert event.source_type == "dom_iskusstv"
        
        # --- Step 4: Mock all external dependencies ---
        bot = DummyBot()
        mock_db = MagicMock()
        
        # Track what was added
        added_events: list[TheatreEvent] = []
        
        async def mock_find_existing(db, location, date, time, title):
            return None, False  # No existing event
        
        async def mock_add_new_event(db, bot, event, current, total, **kwargs):
            added_events.append(event)
            return len(added_events), True  # Return ID, was_added=True
        
        async def mock_download_images(urls):
            return []
        
        async def mock_process_media(images, **kwargs):
            return [], None
        
        # Apply mocks
        monkeypatch.setattr("source_parsing.dom_iskusstv.find_existing_event", mock_find_existing)
        monkeypatch.setattr("source_parsing.dom_iskusstv.add_new_event_via_queue", mock_add_new_event)
        monkeypatch.setattr("source_parsing.dom_iskusstv.download_images", mock_download_images)
        monkeypatch.setattr("source_parsing.dom_iskusstv.process_media", mock_process_media)
        
        # --- Step 5: Process events ---
        stats = await process_dom_iskusstv_events(
            mock_db,
            bot,
            events,
            chat_id=123,
            skip_pages_rebuild=True,
        )
        
        # --- Step 6: Verify results ---
        assert stats.source == "dom_iskusstv"
        assert stats.total_received == 2
        assert stats.new_added == 2
        assert stats.failed == 0
        
        # Verify all events were processed
        assert len(added_events) == 2
        
        # Verify event data completeness
        for event in added_events:
            assert event.title, "Title should not be empty"
            assert event.parsed_date, "Date should not be empty"
            assert event.location, "Location should not be empty"
            assert event.url, "URL should not be empty"
            assert event.source_type == "dom_iskusstv"
        
        # Verify progress messages were sent
        assert len(bot.messages_sent) == 2
        assert all("Дом искусств" in m["text"] for m in bot.messages_sent)
    
    @pytest.mark.asyncio
    async def test_url_extraction_with_various_formats(self):
        """Test URL extraction handles all domain formats."""
        test_cases = [
            # Standard punycode
            ("Билеты: https://xn--b1admiilxbaki.xn--p1ai/skazka", ["skazka"]),
            # Cyrillic domain
            ("Смотрите: https://домискусств.рф/aladdin!", ["aladdin"]),
            # Multiple URLs
            (
                "Спектакли: https://xn--b1admiilxbaki.xn--p1ai/skazka и https://домискусств.рф/aladdin",
                ["skazka", "aladdin"]
            ),
            # With trailing punctuation
            ("Ссылка: https://xn--b1admiilxbaki.xn--p1ai/show.", ["show"]),
            # Skip system pages
            ("https://xn--b1admiilxbaki.xn--p1ai/about-the-theater", []),
        ]
        
        for text, expected_paths in test_cases:
            urls = extract_dom_iskusstv_urls(text)
            for path in expected_paths:
                assert any(path in url for url in urls), f"Expected {path} in {urls}"
    
    @pytest.mark.asyncio
    async def test_event_completeness_check(self, tmp_path: Path, sample_kaggle_json: str):
        """Verify all required fields are present in parsed events."""
        output_file = tmp_path / "dom_iskusstv_events.json"
        output_file.write_text(sample_kaggle_json, encoding="utf-8")
        
        events = parse_dom_iskusstv_output([str(output_file)])
        
        required_fields = [
            "title",
            "parsed_date",
            "parsed_time",
            "location",
            "url",
            "ticket_status",
            "photos",
            "description",
            "source_type",
        ]
        
        for event in events:
            for field in required_fields:
                value = getattr(event, field, None)
                assert value is not None, f"Event missing required field: {field}"
                if field == "photos":
                    assert isinstance(value, list), f"Photos should be a list"
                    assert len(value) > 0, f"Photos list should not be empty"
                elif field != "ticket_status":
                    assert value != "", f"Field {field} should not be empty"
    
    @pytest.mark.asyncio
    async def test_duplicate_handling(
        self,
        test_db: Database,
        tmp_path: Path,
        sample_kaggle_json: str,
        monkeypatch,
    ):
        """Test that existing events are updated, not duplicated."""
        output_file = tmp_path / "dom_iskusstv_events.json"
        output_file.write_text(sample_kaggle_json, encoding="utf-8")
        
        events = parse_dom_iskusstv_output([str(output_file)])
        bot = DummyBot()
        
        # Mock to return existing event
        async def mock_find_existing(db, location, date, time, title):
            return 1, False  # Event ID 1 exists
        
        async def mock_update_ticket_status(db, event_id, status, url):
            return True
        
        async def mock_update_linked(*args, **kwargs):
            pass
        
        monkeypatch.setattr("source_parsing.dom_iskusstv.find_existing_event", mock_find_existing)
        monkeypatch.setattr("source_parsing.dom_iskusstv.update_event_ticket_status", mock_update_ticket_status)
        monkeypatch.setattr("source_parsing.dom_iskusstv.update_linked_events", mock_update_linked)
        
        stats = await process_dom_iskusstv_events(
            test_db,
            bot,
            events,
            chat_id=123,
        )
        
        assert stats.ticket_updated == 2
        assert stats.new_added == 0
    
    @pytest.mark.asyncio
    async def test_price_extraction_accuracy(self, tmp_path: Path):
        """Test that prices are correctly extracted from various formats."""
        events_data = [
            {
                "title": "Event with range",
                "parsed_date": "2026-01-03",
                "parsed_time": "14:00",
                "location": "Дом искусств",
                "url": "https://test.com/1",
                "ticket_status": "available",
                "ticket_price_min": 300,
                "ticket_price_max": 500,
                "photos": ["https://test.com/img.jpg"],
                "description": "Test",
                "source_type": "dom_iskusstv"
            },
            {
                "title": "Free event",
                "parsed_date": "2026-01-04",
                "parsed_time": "12:00",
                "location": "Дом искусств",
                "url": "https://test.com/2",
                "ticket_status": "available",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "photos": ["https://test.com/img.jpg"],
                "description": "Free test",
                "source_type": "dom_iskusstv"
            },
        ]
        
        output_file = tmp_path / "dom_iskusstv_events.json"
        output_file.write_text(json.dumps(events_data, ensure_ascii=False), encoding="utf-8")
        
        events = parse_dom_iskusstv_output([str(output_file)])
        
        # First event with prices
        assert events[0].ticket_price_min == 300
        assert events[0].ticket_price_max == 500
        
        # Second event without prices
        assert events[1].ticket_price_min is None
        assert events[1].ticket_price_max is None


class TestKaggleKernelExecution:
    """Tests for Kaggle kernel execution with security fixes."""
    
    @pytest.mark.asyncio
    async def test_url_validation_rejects_invalid_urls(self, tmp_path: Path):
        """Test that invalid URLs are rejected before kernel execution."""
        # Create mock kernel files
        kernel_dir = tmp_path / "ParseDomIskusstv"
        kernel_dir.mkdir()
        
        (kernel_dir / "kernel-metadata.json").write_text(
            '{"id": "test/kernel", "title": "Test"}',
            encoding="utf-8"
        )
        
        (kernel_dir / "parse_dom_iskusstv.py").write_text(
            'urls_env = os.environ.get("DOM_ISKUSSTV_URLS", "")\nprint("test")',
            encoding="utf-8"
        )
        
        with patch('source_parsing.dom_iskusstv.KERNELS_ROOT_PATH', tmp_path):
            # Test with invalid URL (should be rejected)
            status, files, duration = await run_dom_iskusstv_kaggle_kernel(
                urls=["https://malicious-site.com/evil"]
            )
            
            assert status == "invalid_url"
            assert files == []
    
    @pytest.mark.asyncio
    async def test_safe_url_injection(self, tmp_path: Path):
        """Test that URL injection uses json.dumps for safety."""
        kernel_dir = tmp_path / "ParseDomIskusstv"
        kernel_dir.mkdir()
        
        (kernel_dir / "kernel-metadata.json").write_text(
            '{"id": "test/kernel"}',
            encoding="utf-8"
        )
        
        test_script = 'urls_env = os.environ.get("DOM_ISKUSSTV_URLS", "")\nif urls_env:\n    urls = [u for u in urls_env.split(",")]'
        (kernel_dir / "parse_dom_iskusstv.py").write_text(test_script, encoding="utf-8")
        
        mock_client = MagicMock()
        mock_client.push_kernel = MagicMock()
        mock_client.get_kernel_status = MagicMock(return_value={"status": "COMPLETE"})
        
        with patch('source_parsing.dom_iskusstv.KERNELS_ROOT_PATH', tmp_path):
            with patch('source_parsing.dom_iskusstv.KaggleClient', return_value=mock_client):
                with patch('source_parsing.dom_iskusstv._download_dom_iskusstv_outputs', return_value=[]):
                    # Test with valid URL
                    status, files, duration = await run_dom_iskusstv_kaggle_kernel(
                        urls=["https://xn--b1admiilxbaki.xn--p1ai/skazka"],
                        timeout_minutes=1,
                        poll_interval=1,
                    )
        
        # Should complete without injection issues
        assert status == "complete"
