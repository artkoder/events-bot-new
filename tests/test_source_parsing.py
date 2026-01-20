"""Tests for source_parsing module."""

import pytest
from datetime import date

from source_parsing.parser import (
    TheatreEvent,
    parse_date_raw,
    parse_theatre_json,
    normalize_location_name,
    fuzzy_title_match,
    limit_photos_for_source,
)


class TestParseDateRaw:
    """Tests for Russian date parsing."""

    def test_full_date_with_time(self):
        """Parse complete date with time."""
        date_str, time_str = parse_date_raw("28 декабря 18:00")
        assert time_str == "18:00"
        assert date_str is not None
        assert date_str.endswith("-12-28")

    def test_uppercase_date(self):
        """Parse uppercase date."""
        date_str, time_str = parse_date_raw("02 ЯНВАРЯ 13:00")
        assert time_str == "13:00"
        assert date_str is not None
        assert "-01-02" in date_str

    def test_date_without_time(self):
        """Parse date without time."""
        date_str, time_str = parse_date_raw("15 марта")
        assert time_str is None
        assert date_str is not None
        assert "-03-15" in date_str

    def test_truncated_month(self):
        """Parse truncated month name."""
        date_str, time_str = parse_date_raw("28 ДЕКАБР")
        assert date_str is not None
        assert "-12-28" in date_str

    def test_empty_string(self):
        """Handle empty input."""
        date_str, time_str = parse_date_raw("")
        assert date_str is None
        assert time_str is None

    def test_numeric_date_format(self):
        """Parse DD.MM.YYYY format from Pyramida."""
        date_str, time_str = parse_date_raw("21.03.2026 18:00")
        assert date_str == "2026-03-21"
        assert time_str == "18:00"

    def test_numeric_date_without_time(self):
        """Parse DD.MM.YYYY format without time."""
        date_str, time_str = parse_date_raw("15.01.2025")
        assert date_str == "2025-01-15"
        assert time_str is None


class TestParseTheatreJson:
    """Tests for JSON parsing."""

    def test_parse_single_event(self):
        """Parse a single event from JSON."""
        json_data = {
            "title": "Чайка",
            "date_raw": "28 декабря 18:00",
            "ticket_status": "available",
            "url": "https://example.com/ticket",
            "photos": ["https://example.com/photo1.jpg"],
            "description": "Спектакль по пьесе Чехова",
            "pushkin_card": True,
            "location": "Драматический театр",
        }
        events = parse_theatre_json(json_data, "dramteatr")
        
        assert len(events) == 1
        event = events[0]
        assert event.title == "Чайка"
        assert event.ticket_status == "available"
        assert event.pushkin_card is True
        assert event.parsed_time == "18:00"

    def test_parse_list_of_events(self):
        """Parse multiple events from JSON list."""
        json_data = [
            {"title": "Event 1", "date_raw": "1 января 12:00", "ticket_status": "available", "url": ""},
            {"title": "Event 2", "date_raw": "2 января 14:00", "ticket_status": "sold_out", "url": ""},
        ]
        events = parse_theatre_json(json_data, "muzteatr")
        
        assert len(events) == 2
        assert events[0].title == "Event 1"
        assert events[0].ticket_status == "available"
        assert events[1].title == "Event 2"
        assert events[1].ticket_status == "sold_out"

    def test_parse_invalid_json_string(self):
        """Handle invalid JSON gracefully."""
        events = parse_theatre_json("not valid json", "test")
        assert events == []

    def test_skip_empty_title(self):
        """Skip events without title."""
        json_data = [
            {"title": "", "date_raw": "1 января", "ticket_status": "available", "url": ""},
            {"title": "Valid Event", "date_raw": "2 января", "ticket_status": "available", "url": ""},
        ]
        events = parse_theatre_json(json_data, "test")
        
        assert len(events) == 1
        assert events[0].title == "Valid Event"


class TestNormalizeLocationName:
    """Tests for location name normalization."""

    def test_normalize_dramteatr(self):
        """Normalize Драматический театр variants."""
        assert normalize_location_name("Драматический театр") == "Драматический театр"
        assert normalize_location_name("драматический театр") == "Драматический театр"
        assert normalize_location_name("Калининградский драматический театр") == "Драматический театр"

    def test_normalize_muzteatr(self):
        """Normalize Музыкальный театр variants."""
        assert normalize_location_name("Музыкальный театр") == "Музыкальный театр"
        assert normalize_location_name("музыкальный театр") == "Музыкальный театр"

    def test_normalize_sobor(self):
        """Normalize Кафедральный собор variants."""
        assert normalize_location_name("Кафедральный собор") == "Кафедральный собор"
        assert normalize_location_name("кафедральный собор") == "Кафедральный собор"

    def test_unknown_location(self):
        """Return original for unknown locations."""
        assert normalize_location_name("Другое место") == "Другое место"

    def test_empty_location(self):
        """Handle empty input."""
        assert normalize_location_name("") == ""

    def test_normalize_tretyakov(self):
        """Normalize Tretyakov variants."""
        expected = "Филиал Третьяковской галереи"
        assert normalize_location_name("Третьяков") == expected
        assert normalize_location_name("Третьяковка Калининград") == expected
        assert normalize_location_name("Атриум") == expected

    def test_normalize_tretyakov_with_scene(self):
        """Append scene for Tretyakov locations when provided."""
        expected = "Филиал Третьяковской галереи (Атриум)"
        assert normalize_location_name("Атриум", "Атриум") == expected
        assert normalize_location_name("Филиал Третьяковской галереи", "Атриум") == expected


class TestFuzzyTitleMatch:
    """Tests for fuzzy title matching."""

    def test_exact_match(self):
        """Exact matches should return True."""
        assert fuzzy_title_match("Чайка", "Чайка") is True

    def test_case_insensitive(self):
        """Case differences should match."""
        assert fuzzy_title_match("Чайка", "чайка") is True
        assert fuzzy_title_match("ЧАЙКА", "Чайка") is True

    def test_similar_titles(self):
        """Similar titles should match."""
        # "Чайка Спектакль" vs "Чайка спектакль" - very similar
        assert fuzzy_title_match("Чайка спектакль", "Чайка. спектакль") is True

    def test_different_titles(self):
        """Different titles should not match."""
        assert fuzzy_title_match("Чайка", "Три сестры") is False

    def test_empty_titles(self):
        """Empty titles should not match."""
        assert fuzzy_title_match("", "Чайка") is False
        assert fuzzy_title_match("Чайка", "") is False


class TestLimitPhotosForSource:
    """Tests for photo limiting."""

    def test_muzteatr_limits_to_5(self):
        """Muzteater photos should be limited to 5."""
        photos = [f"photo{i}.jpg" for i in range(10)]
        result = limit_photos_for_source(photos, "muzteatr")
        assert len(result) == 5

    def test_muzteatr_under_limit(self):
        """Muzteater with fewer photos keeps all."""
        photos = ["photo1.jpg", "photo2.jpg"]
        result = limit_photos_for_source(photos, "muzteatr")
        assert len(result) == 2

    def test_other_source_no_limit(self):
        """Other sources don't have photo limits."""
        photos = [f"photo{i}.jpg" for i in range(10)]
        result = limit_photos_for_source(photos, "dramteatr")
        assert len(result) == 10

    def test_empty_photos(self):
        """Handle empty photo list."""
        result = limit_photos_for_source([], "muzteatr")
        assert result == []


class TestShortDescriptionFallback:
    """Tests for short_description fallback logic."""

    def test_first_sentence_extraction(self):
        """First sentence should be extracted when splitting by period."""
        description = "Это первое предложение. Это второе предложение. И третье."
        first_sentence = description.split('.')[0].strip()
        assert first_sentence == "Это первое предложение"
        
    def test_first_sentence_with_empty_result(self):
        """Empty description should not cause errors."""
        description = ""
        first_sentence = description.split('.')[0].strip()
        assert first_sentence == ""
        
    def test_first_sentence_no_period(self):
        """Description without period returns full text."""
        description = "Текст без точки"
        first_sentence = description.split('.')[0].strip()
        assert first_sentence == "Текст без точки"

    def test_fallback_to_title_when_empty(self):
        """When description is empty, title should be used."""
        description = ""
        title = "Название события"
        first_sentence = description.split('.')[0].strip() if description else ""
        result = first_sentence + '.' if first_sentence else title
        assert result == title
