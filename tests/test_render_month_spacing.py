from datetime import date

import main
from telegraph.utils import nodes_to_html
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
    assert "<p>\u200B</p><h4>" in html


def test_telegraph_br_no_span():
    html = nodes_to_html(main.telegraph_br())
    assert html == "<p>\u200B</p>"
    assert "<span" not in html


def test_lint_preserves_zwsp():
    assert main.lint_telegraph_html("<p>&#8203;</p>") == "<p>&#8203;</p>"


def test_event_to_nodes_ends_with_blank_paragraph():
    event = make_event("C", "2025-01-15", "12:00")
    nodes = main.event_to_nodes(event)
    assert nodes[-1] == main.telegraph_br()[0]
    assert nodes_to_html(nodes).endswith("<p>\u200B</p>")
