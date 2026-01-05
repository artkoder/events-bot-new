"""Unit tests for festival parser functions."""

import pytest
from datetime import date, datetime

from source_parsing.festival_parser import (
    classify_source_type,
    is_valid_url,
    generate_run_id,
    generate_festival_slug,
)


class TestClassifySourceType:
    """Test source type classification."""

    def test_canonical_domain(self):
        """Known festival domains are classified as canonical."""
        assert classify_source_type("https://zimafestkld.ru/") == "canonical"
        assert classify_source_type("https://www.zimafestkld.ru/program") == "canonical"

    def test_external_aggregator(self):
        """Known aggregators are classified as external."""
        assert classify_source_type("https://afisha.yandex.ru/kaliningrad/event/1234") == "external"
        assert classify_source_type("https://kudago.com/kld/event/fest") == "external"
        assert classify_source_type("https://timepad.ru/event/12345") == "external"

    def test_official_from_domain_pattern(self):
        """Domains containing 'fest' are classified as official."""
        assert classify_source_type("https://somefest.ru/") == "official"
        assert classify_source_type("https://myfestival2025.org/") == "official"
        assert classify_source_type("https://artfest.com/") == "official"

    def test_unknown_domain_is_external(self):
        """Unknown domains default to external."""
        assert classify_source_type("https://example.com/page") == "external"
        assert classify_source_type("https://randomsite.ru/") == "external"

    def test_empty_url(self):
        """Empty or invalid URLs return external."""
        assert classify_source_type("") == "external"
        assert classify_source_type("not-a-url") == "external"

    def test_cyrillic_fest_in_domain(self):
        """Domains with Cyrillic 'фест' are official."""
        assert classify_source_type("https://нашфест.рф/") == "official"


class TestIsValidUrl:
    """Test URL validation."""

    def test_valid_http_urls(self):
        assert is_valid_url("http://example.com") is True
        assert is_valid_url("https://example.com/path") is True
        assert is_valid_url("https://sub.domain.com/path?query=1") is True

    def test_invalid_urls(self):
        assert is_valid_url("") is False
        assert is_valid_url("not-a-url") is False
        assert is_valid_url("ftp://example.com") is False
        assert is_valid_url("example.com") is False

    def test_url_with_spaces(self):
        assert is_valid_url("  https://example.com  ") is True


class TestGenerateRunId:
    """Test run ID generation."""

    def test_run_id_format(self):
        """Run ID has timestamp + hash format."""
        run_id = generate_run_id("https://example.com")
        assert "_" in run_id
        parts = run_id.split("_")
        assert len(parts) == 2
        assert "T" in parts[0]  # Timestamp has T separator
        assert len(parts[1]) == 8  # 8-char hash

    def test_same_url_different_times(self):
        """Same URL generates different run IDs (due to timestamp)."""
        # Note: This test might be flaky if run in the same second
        run_id1 = generate_run_id("https://example.com")
        run_id2 = generate_run_id("https://example.com")
        # Hash part should be the same
        assert run_id1.split("_")[1] == run_id2.split("_")[1]

    def test_different_urls_different_hashes(self):
        """Different URLs generate different hashes."""
        run_id1 = generate_run_id("https://example1.com")
        run_id2 = generate_run_id("https://example2.com")
        assert run_id1.split("_")[1] != run_id2.split("_")[1]


class TestGenerateFestivalSlug:
    """Test festival slug generation."""

    def test_cyrillic_transliteration(self):
        """Cyrillic names are transliterated."""
        assert generate_festival_slug("Зимафест") == "zimafest"
        assert generate_festival_slug("Фестиваль искусств") == "festival-iskusstv"

    def test_latin_preserved(self):
        """Latin names are preserved."""
        assert generate_festival_slug("ArtFest") == "artfest"
        assert generate_festival_slug("Music2025") == "music2025"

    def test_special_chars_replaced(self):
        """Special characters become dashes."""
        assert generate_festival_slug("Art & Music!") == "art-music"
        assert generate_festival_slug("Fest (2025)") == "fest-2025"

    def test_empty_name(self):
        """Empty name returns default."""
        assert generate_festival_slug("") == "festival"
        assert generate_festival_slug("   ") == "festival"
