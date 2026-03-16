"""Date formatting and implicit-year resolution utilities.

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


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except (TypeError, ValueError):
        return None


def resolve_implicit_year_date(
    day: int,
    month: int,
    *,
    anchor_date: date,
    recent_past_days: int = 0,
) -> Optional[date]:
    """Resolve a day+month mention relative to the source/post anchor date."""
    try:
        recent_past_days = max(int(recent_past_days or 0), 0)
    except (TypeError, ValueError):
        recent_past_days = 0

    candidate = _safe_date(int(anchor_date.year), int(month), int(day))
    if candidate is None:
        return None
    if candidate >= anchor_date:
        return candidate
    if (anchor_date - candidate).days <= recent_past_days:
        return candidate
    return _safe_date(int(anchor_date.year) + 1, int(month), int(day)) or candidate


def normalize_implicit_iso_date_to_anchor(
    iso_date: str | date | datetime | None,
    *,
    anchor_date: date,
    recent_past_days: int = 0,
) -> Optional[str]:
    """Re-anchor an ISO date that came from a source without an explicit year."""
    if iso_date is None:
        return None
    if isinstance(iso_date, datetime):
        parsed = iso_date.date()
    elif isinstance(iso_date, date):
        parsed = iso_date
    else:
        raw = str(iso_date).split("..", 1)[0].strip()
        if not raw:
            return None
        try:
            parsed = date.fromisoformat(raw)
        except (TypeError, ValueError):
            return None
    resolved = resolve_implicit_year_date(
        parsed.day,
        parsed.month,
        anchor_date=anchor_date,
        recent_past_days=recent_past_days,
    )
    return resolved.isoformat() if resolved else None


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
