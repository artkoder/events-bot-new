from datetime import date

import main
from markup import DAY_START, DAY_END, PERM_START, PERM_END


def test_ensure_day_markers_once():
    html = f"<p>x</p>{PERM_START}{PERM_END}"
    d = date(2025, 8, 15)
    updated, changed = main.ensure_day_markers(html, d)
    assert changed is True
    assert DAY_START(d) in updated and DAY_END(d) in updated
    assert updated.index(DAY_START(d)) < updated.index(PERM_START)
    again, changed2 = main.ensure_day_markers(updated, d)
    assert changed2 is False
    assert again == updated
