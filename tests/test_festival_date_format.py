"""Unit tests for date formatting utilities."""

import pytest
from datetime import date, datetime

from source_parsing.date_utils import (
    format_date_for_display,
    format_date_range_for_display,
)


class TestFormatDateForDisplay:
    """Test date formatting with Russian month names."""

    def test_current_year_omits_year(self):
        """Dates in current year omit the year."""
        current_year = date.today().year
        result = format_date_for_display(f"{current_year}-06-15", current_year)
        assert result == "15 июня"

    def test_different_year_includes_year(self):
        """Dates in different year include the year."""
        result = format_date_for_display("2024-06-15", 2025)
        assert result == "15 июня 2024"

    def test_all_months(self):
        """All months are formatted correctly."""
        expected = [
            ("01", "января"),
            ("02", "февраля"),
            ("03", "марта"),
            ("04", "апреля"),
            ("05", "мая"),
            ("06", "июня"),
            ("07", "июля"),
            ("08", "августа"),
            ("09", "сентября"),
            ("10", "октября"),
            ("11", "ноября"),
            ("12", "декабря"),
        ]
        for month_num, month_name in expected:
            result = format_date_for_display(f"2025-{month_num}-15", 2025)
            assert month_name in result, f"Month {month_num} should contain {month_name}"

    def test_date_object_input(self):
        """Accepts date objects."""
        d = date(2025, 3, 20)
        result = format_date_for_display(d, 2025)
        assert result == "20 марта"

    def test_datetime_object_input(self):
        """Accepts datetime objects."""
        dt = datetime(2025, 3, 20, 15, 30)
        result = format_date_for_display(dt, 2025)
        assert result == "20 марта"

    def test_iso_with_time(self):
        """Handles ISO strings with time component."""
        result = format_date_for_display("2025-07-04T14:30:00Z", 2025)
        assert result == "4 июля"

    def test_malformed_date_returns_original(self):
        """Malformed dates return original string."""
        result = format_date_for_display("not-a-date", 2025)
        assert result == "not-a-date"

    def test_uses_current_year_by_default(self):
        """Uses current year if reference_year not provided."""
        current_year = date.today().year
        result = format_date_for_display(f"{current_year}-08-15")
        assert result == "15 августа"


class TestFormatDateRangeForDisplay:
    """Test date range formatting."""

    def test_same_day_shows_single_date(self):
        """Same start and end shows single date."""
        result = format_date_range_for_display("2025-06-15", "2025-06-15", 2025)
        assert result == "15 июня"

    def test_different_days_shows_range(self):
        """Different days show range with dash."""
        result = format_date_range_for_display("2025-06-15", "2025-06-20", 2025)
        assert result == "15 июня - 20 июня"

    def test_no_end_date(self):
        """Missing end date shows only start."""
        result = format_date_range_for_display("2025-06-15", None, 2025)
        assert result == "15 июня"

    def test_no_start_date(self):
        """Missing start date returns empty string."""
        result = format_date_range_for_display(None, "2025-06-20", 2025)
        assert result == ""

    def test_cross_year_range(self):
        """Range crossing years includes years."""
        result = format_date_range_for_display("2024-12-28", "2025-01-05", 2025)
        assert result == "28 декабря 2024 - 5 января"
