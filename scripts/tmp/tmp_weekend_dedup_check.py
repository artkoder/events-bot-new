"""Test weekend header duplication fix"""
import pytest
from datetime import date
from telegraph.utils import html_to_nodes, nodes_to_html
from sections import dedup_same_date


def test_dedup_removes_weekend_header_on_duplicate():
    """Test that dedup_same_date removes weekend headers from duplicate sections."""
    # Simulate HTML with duplicate Saturday sections
    html = """
    <p>​</p>
    <h3>🟥🟥🟥 суббота 🟥🟥🟥</h3>
    <h3>🟥🟥🟥 28 декабря 🟥🟥🟥</h3>
    <p>​</p>
    <h4>Event A</h4>
    <p>Description A</p>
    <p>​</p>
    <h3>🟥🟥🟥 суббота 🟥🟥🟥</h3>
    <h3>🟥🟥🟥 28 декабря 🟥🟥🟥</h3>
    <p>​</p>
    <h4>Event B</h4>
    <p>Description B</p>
    <p>​</p>
    <hr/>
    """
    
    nodes = html_to_nodes(html)
    target = date(2025, 12, 28)
    
    # Before dedup: should have 2 "суббота" headers and 2 date headers
    html_before = nodes_to_html(nodes)
    assert html_before.count("суббота") == 2
    assert html_before.count("28 декабря") == 2
    
    # Apply dedup
    result_nodes, removed_count = dedup_same_date(nodes, target)
    
    # After dedup: should have only 1 "суббота" header and 1 date header
    html_after = nodes_to_html(result_nodes)
    assert html_after.count("суббота") == 1, f"Expected 1 'суббота', got {html_after.count('суббота')}"
    assert html_after.count("28 декабря") == 1, f"Expected 1 '28 декабря', got {html_after.count('28 декабря')}"
    assert removed_count == 1
    
    # First event should remain
    assert "Event A" in html_after
    # Second event should be removed with the duplicate
    assert "Event B" not in html_after
    
    print("✅ Test passed: Weekend header duplication fixed!")
    print(f"Before: {html_before.count('суббота')} Saturday headers, {html_before.count('28 декабря')} date headers")
    print(f"After: {html_after.count('суббота')} Saturday header, {html_after.count('28 декабря')} date header")


if __name__ == "__main__":
    test_dedup_removes_weekend_header_on_duplicate()
