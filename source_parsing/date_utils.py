"""Date formatting utilities for festival and event pages.

Provides unified date formatting following project standards:
- "DD месяц" when year matches current year
- "DD месяц YYYY" when year differs from current year
"""

from datetime import date, datetime
from typing import Optional

# Russian month names in genitive case (для дат: "1 января", "15 февраля")
MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]


def format_date_for_display(
    iso_date: str | date | datetime,
    reference_year: Optional[int] = None,
) -> str:
    """Format date as 'DD месяц' or 'DD месяц YYYY'.
    
    Args:
        iso_date: Date in ISO format (YYYY-MM-DD) or date/datetime object
        reference_year: Year to compare against. If None, uses current year.
        
    Returns:
        Formatted date string like "12 июня" or "12 июня 2025"
        
    Examples:
        >>> format_date_for_display("2025-06-12")  # If current year is 2025
        '12 июня'
        >>> format_date_for_display("2024-06-12")  # If current year is 2025
        '12 июня 2024'
    """
    if isinstance(iso_date, datetime):
        dt = iso_date.date()
    elif isinstance(iso_date, date):
        dt = iso_date
    else:
        # Parse ISO string
        try:
            if "T" in iso_date:
                dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00")).date()
            else:
                dt = date.fromisoformat(iso_date[:10])
        except (ValueError, TypeError):
            return str(iso_date)  # Fallback for malformed dates
    
    if reference_year is None:
        reference_year = date.today().year
    
    day = dt.day
    month_name = MONTHS_GENITIVE[dt.month - 1]
    
    if dt.year == reference_year:
        return f"{day} {month_name}"
    else:
        return f"{day} {month_name} {dt.year}"


def format_date_range_for_display(
    start_date: str | date | datetime | None,
    end_date: str | date | datetime | None,
    reference_year: Optional[int] = None,
) -> str:
    """Format date range as "DD месяц - DD месяц" or "DD месяц YYYY - DD месяц YYYY".
    
    Args:
        start_date: Start date in ISO format or date object
        end_date: End date in ISO format or date object (can be None)
        reference_year: Year to compare against. If None, uses current year.
        
    Returns:
        Formatted date range string
        
    Examples:
        >>> format_date_range_for_display("2025-06-12", "2025-06-15")
        '12 июня - 15 июня'
        >>> format_date_range_for_display("2025-06-12", None)
        '12 июня'
    """
    if start_date is None:
        return ""
    
    start_str = format_date_for_display(start_date, reference_year)
    
    if end_date is None:
        return start_str
    
    end_str = format_date_for_display(end_date, reference_year)
    
    if start_str == end_str:
        return start_str
    
    return f"{start_str} - {end_str}"
