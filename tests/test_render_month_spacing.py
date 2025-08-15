from datetime import date

import main
from models import Event


def make_event(title: str, d: str, time: str) -> Event:
    return Event(
        title=title,
        description="desc",
        date=d,
        time=time,
        location_name="loc",
        source_text="src",
    )


def test_render_month_day_section_has_blank_lines():
    events = [
        make_event("A", "2025-01-15", "12:00"),
        make_event("B", "2025-01-15", "13:00"),
    ]
    html = main.render_month_day_section(date(2025, 1, 15), events)
    assert "<p>\u00A0</p><h4>" in html


def test_telegraph_br_no_span():
    from telegraph.utils import nodes_to_html

    html = nodes_to_html(main.telegraph_br())
    assert html == "<p>\u00A0</p>"
    assert "<span" not in html


def test_lint_preserves_nbsp():
    assert main.lint_telegraph_html("<p>&nbsp;</p>") == "<p>&nbsp;</p>"
