"""Tests for Pyramida event extraction module."""

import pytest
from source_parsing.pyramida import extract_pyramida_urls


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
