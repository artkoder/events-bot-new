"""Tests for Dom Iskusstv event extraction module."""

import pytest
import json
import tempfile
from pathlib import Path
from source_parsing.dom_iskusstv import (
    extract_dom_iskusstv_urls, 
    parse_dom_iskusstv_output, 
    parse_price_string,
)


class TestExtractDomIskusstvUrls:
    """Tests for extract_dom_iskusstv_urls function."""

    def test_extract_punycode_url(self):
        """Extract punycode URL."""
        text = "Билеты на https://xn--b1admiilxbaki.xn--p1ai/skazka"
        result = extract_dom_iskusstv_urls(text)
        assert result == ["https://xn--b1admiilxbaki.xn--p1ai/skazka"]

    def test_extract_cyrillic_url(self):
        """Extract cyrillic domain URL."""
        text = "Смотрите https://домискусств.рф/aladdin"
        result = extract_dom_iskusstv_urls(text)
        assert result == ["https://xn--b1admiilxbaki.xn--p1ai/aladdin"]

    def test_extract_multiple_urls(self):
        """Extract multiple special project URLs."""
        text = """
        Первый проект: https://xn--b1admiilxbaki.xn--p1ai/skazka
        Второй проект: https://xn--b1admiilxbaki.xn--p1ai/aladdin
        """
        result = extract_dom_iskusstv_urls(text)
        assert len(result) == 2
        assert "https://xn--b1admiilxbaki.xn--p1ai/skazka" in result
        assert "https://xn--b1admiilxbaki.xn--p1ai/aladdin" in result

    def test_skip_non_project_urls(self):
        """Skip non-project paths like /posters, /news1."""
        text = """
        https://xn--b1admiilxbaki.xn--p1ai/posters
        https://xn--b1admiilxbaki.xn--p1ai/news1
        https://xn--b1admiilxbaki.xn--p1ai/contacts
        """
        result = extract_dom_iskusstv_urls(text)
        assert result == []

    def test_skip_about_pages(self):
        """Skip about/info pages."""
        text = """
        https://xn--b1admiilxbaki.xn--p1ai/about-the-theater
        https://xn--b1admiilxbaki.xn--p1ai/faq
        """
        result = extract_dom_iskusstv_urls(text)
        assert result == []

    def test_extract_only_project_urls(self):
        """Extract only project URLs, skip info pages."""
        text = """
        https://xn--b1admiilxbaki.xn--p1ai/skazka
        https://xn--b1admiilxbaki.xn--p1ai/posters
        https://xn--b1admiilxbaki.xn--p1ai/aladdin
        """
        result = extract_dom_iskusstv_urls(text)
        assert len(result) == 2
        assert "https://xn--b1admiilxbaki.xn--p1ai/skazka" in result
        assert "https://xn--b1admiilxbaki.xn--p1ai/aladdin" in result

    def test_deduplicate_urls(self):
        """Duplicate URLs should be removed."""
        text = """
        https://xn--b1admiilxbaki.xn--p1ai/skazka
        Again: https://xn--b1admiilxbaki.xn--p1ai/skazka
        """
        result = extract_dom_iskusstv_urls(text)
        assert result == ["https://xn--b1admiilxbaki.xn--p1ai/skazka"]

    def test_no_dom_iskusstv_urls(self):
        """Return empty list when no Dom Iskusstv URLs found."""
        text = "Билеты на pyramida.info/tickets/12345"
        result = extract_dom_iskusstv_urls(text)
        assert result == []

    def test_empty_text(self):
        """Return empty list for empty input."""
        assert extract_dom_iskusstv_urls("") == []
        assert extract_dom_iskusstv_urls(None) == []

    def test_url_with_trailing_punctuation(self):
        """Trailing punctuation should be cleaned."""
        text = "Ссылка: https://xn--b1admiilxbaki.xn--p1ai/skazka."
        result = extract_dom_iskusstv_urls(text)
        assert result == ["https://xn--b1admiilxbaki.xn--p1ai/skazka"]


class TestParsePriceString:
    """Tests for price parsing helper."""
    
    def test_simple_price(self):
        assert parse_price_string("500 ₽") == (500, 500)
        assert parse_price_string("1000 руб") == (1000, 1000)
        assert parse_price_string("1500") == (1500, 1500)
    
    def test_price_range(self):
        assert parse_price_string("500 - 1000 ₽") == (500, 1000)
        assert parse_price_string("500-1000") == (500, 1000)
    
    def test_price_from(self):
        assert parse_price_string("от 500 ₽") == (500, None)
        assert parse_price_string("от 1000") == (1000, None)
    
    def test_free(self):
        assert parse_price_string("Бесплатно") == (0, 0)
        assert parse_price_string("Вход свободный") == (0, 0)
        
    def test_invalid(self):
        assert parse_price_string("") == (None, None)
        assert parse_price_string("invalid") == (None, None)


class TestParseDomIskusstvOutput:
    """Tests for parse_dom_iskusstv_output function."""
    
    def test_parse_complete_event(self):
        """Test parsing JSON with all fields."""
        data = [
            {
                "title": "Сказки на ночь",
                "date_raw": "3 января 14:00",
                "parsed_date": "2026-01-03",
                "parsed_time": "14:00",
                "location": "Дом искусств",
                "url": "https://xn--b1admiilxbaki.xn--p1ai/?unifd-performance-id=832",
                "ticket_status": "available",
                "ticket_price_min": 500,
                "ticket_price_max": 1000,
                "photos": ["https://example.com/img.jpg"],
                "description": "Музыкальный спектакль",
                "age_restriction": "6+",
                "source_type": "dom_iskusstv"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='dom_iskusstv_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_dom_iskusstv_output([temp_path])
            assert len(events) == 1
            event = events[0]
            
            assert event.title == "Сказки на ночь"
            assert event.parsed_date == "2026-01-03"
            assert event.parsed_time == "14:00"
            assert event.location == "Дом искусств"
            assert event.ticket_status == "available"
            assert event.ticket_price_min == 500
            assert event.ticket_price_max == 1000
            assert event.age_restriction == "6+"
            assert event.source_type == "dom_iskusstv"
            
        finally:
            Path(temp_path).unlink()

    def test_parse_multiple_events(self):
        """Test parsing multiple events from same JSON."""
        data = [
            {
                "title": "Сказки на ночь",
                "date_raw": "3 января 14:00",
                "parsed_date": "2026-01-03",
                "parsed_time": "14:00",
                "location": "Дом искусств",
                "url": "https://example.com/1",
                "ticket_status": "available"
            },
            {
                "title": "Сказки на ночь",
                "date_raw": "4 января 14:00",
                "parsed_date": "2026-01-04",
                "parsed_time": "14:00",
                "location": "Дом искусств",
                "url": "https://example.com/2",
                "ticket_status": "available"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='dom_iskusstv_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_dom_iskusstv_output([temp_path])
            assert len(events) == 2
            
            # Events should have different dates
            dates = {e.parsed_date for e in events}
            assert dates == {"2026-01-03", "2026-01-04"}
            
        finally:
            Path(temp_path).unlink()

    def test_default_location(self):
        """Test that default location is 'Дом искусств'."""
        data = [
            {
                "title": "Тест",
                "date_raw": "5 января 15:00",
                "parsed_date": "2026-01-05",
                "parsed_time": "15:00",
                "url": "https://example.com",
                "ticket_status": "unknown"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='dom_iskusstv_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_dom_iskusstv_output([temp_path])
            assert len(events) == 1
            assert events[0].location == "Дом искусств"
            
        finally:
            Path(temp_path).unlink()

    def test_skip_event_without_title(self):
        """Skip events without title."""
        data = [
            {
                "title": "",
                "date_raw": "5 января 15:00",
                "url": "https://example.com"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='dom_iskusstv_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_dom_iskusstv_output([temp_path])
            assert len(events) == 0
            
        finally:
            Path(temp_path).unlink()
