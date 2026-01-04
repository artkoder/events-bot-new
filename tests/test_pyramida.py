"""Tests for Pyramida event extraction module."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from source_parsing.pyramida import extract_pyramida_urls, parse_pyramida_output, parse_price_string
from source_parsing.parser import TheatreEvent

class TestExtractPyramidaUrls:
    """Tests for extract_pyramida_urls function."""

    def test_extract_single_url(self):
        """Extract a single Pyramida URL from text."""
        text = "Билеты на https://pyramida.info/tickets/event_12345"
        result = extract_pyramida_urls(text)
        assert result == ["https://pyramida.info/tickets/event_12345"]

    def test_extract_multiple_urls(self):
        """Extract multiple Pyramida URLs from text."""
        text = """
        Первое событие: https://pyramida.info/tickets/event_111
        Второе событие: https://pyramida.info/tickets/event_222
        """
        result = extract_pyramida_urls(text)
        assert len(result) == 2
        assert "https://pyramida.info/tickets/event_111" in result
        assert "https://pyramida.info/tickets/event_222" in result

    def test_extract_with_www(self):
        """Extract URL with www prefix."""
        text = "https://www.pyramida.info/tickets/show_123"
        result = extract_pyramida_urls(text)
        assert result == ["https://www.pyramida.info/tickets/show_123"]

    def test_extract_http_scheme(self):
        """Extract URL with http scheme."""
        text = "http://pyramida.info/tickets/event_456"
        result = extract_pyramida_urls(text)
        assert result == ["http://pyramida.info/tickets/event_456"]

    def test_extract_urls_with_trailing_punctuation(self):
        """Trailing punctuation should be cleaned."""
        text = "Ссылка: https://pyramida.info/tickets/event_789."
        result = extract_pyramida_urls(text)
        assert result == ["https://pyramida.info/tickets/event_789"]

    def test_deduplicate_urls(self):
        """Duplicate URLs should be removed."""
        text = """
        https://pyramida.info/tickets/same_event
        Опять: https://pyramida.info/tickets/same_event
        """
        result = extract_pyramida_urls(text)
        assert result == ["https://pyramida.info/tickets/same_event"]

    def test_no_pyramida_urls(self):
        """Return empty list when no Pyramida URLs found."""
        text = "Билеты на concert.ru/tickets/12345"
        result = extract_pyramida_urls(text)
        assert result == []

    def test_empty_text(self):
        """Return empty list for empty input."""
        assert extract_pyramida_urls("") == []
        assert extract_pyramida_urls(None) == []

    def test_ignore_non_ticket_urls(self):
        """Ignore Pyramida URLs that don't contain /tickets/."""
        text = "https://pyramida.info/about"
        result = extract_pyramida_urls(text)
        assert result == []

    def test_complex_url_slug(self):
        """Handle complex URL slugs with underscores and numbers."""
        text = "https://pyramida.info/tickets/novogodnee-nastroenie-ot-tantsy_54151730"
        result = extract_pyramida_urls(text)
        assert result == ["https://pyramida.info/tickets/novogodnee-nastroenie-ot-tantsy_54151730"]

    def test_urls_in_parentheses(self):
        """Extract URL from inside parentheses."""
        text = "(https://pyramida.info/tickets/event_abc)"
        result = extract_pyramida_urls(text)
        # URL should be extracted without closing paren
        assert len(result) == 1
        assert result[0].startswith("https://pyramida.info/tickets/event_abc")

    def test_urls_in_markdown(self):
        """Extract URL from markdown-style link."""
        text = "[Купить билеты](https://pyramida.info/tickets/event_xyz)"
        result = extract_pyramida_urls(text)
        assert len(result) == 1
        # The closing paren is part of the slug in this case
        assert "pyramida.info/tickets/event_xyz" in result[0]


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
        assert parse_price_string("Вход свободный") == (0, 0) # Fallback to 0 if detected as free
        
    def test_invalid(self):
        assert parse_price_string("") == (None, None)
        assert parse_price_string("invalid") == (None, None)


class TestParsePyramidaOutput:
    """Tests for parse_pyramida_output function."""
    
    def test_parse_with_price(self):
        """Test parsing JSON with price field."""
        data = [
            {
                "title": "Тестовое событие",
                "date_raw": "01.01.2026 18:00",
                "location": "Театр",
                "price": "500 ₽",
                "age_restriction": "12+",
                "ticket_status": "unknown",
                "url": "https://pyramida.info/ticket",
                "image_url": "https://example.com/img.jpg",
                "description": "Desc"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='pyramida_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_pyramida_output([temp_path])
            assert len(events) == 1
            event = events[0]
            
            assert event.title == "Тестовое событие"
            assert event.ticket_price_min == 500
            assert event.ticket_price_max == 500
            # Should be updated to available because price exists
            assert event.ticket_status == "available"
            
        finally:
            Path(temp_path).unlink()
            
    def test_parse_without_price(self):
        """Test parsing JSON without price."""
        data = [
            {
                "title": "Без цены",
                "date_raw": "01.01.2026 18:00",
                "url": "https://pyramida.info/ticket",
                "ticket_status": "unknown"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='pyramida_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_pyramida_output([temp_path])
            assert len(events) == 1
            event = events[0]
            
            assert event.ticket_price_min is None
            assert event.ticket_status == "unknown"
            
        finally:
            Path(temp_path).unlink()

    def test_parse_with_range_price(self):
        """Test parsing JSON with price range."""
        data = [
            {
                "title": "Диапазон",
                "date_raw": "01.01.2026 18:00",
                "price": "500 - 1500 ₽",
                "url": "https://pyramida.info/ticket",
                "ticket_status": "unknown"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='pyramida_', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
            
        try:
            events = parse_pyramida_output([temp_path])
            assert len(events) == 1
            event = events[0]
            
            assert event.ticket_price_min == 500
            assert event.ticket_price_max == 1500
            assert event.ticket_status == "available"
            
        finally:
            Path(temp_path).unlink()
