from datetime import date

from sections import parse_month_sections


def test_parse_month_sections_nbsp_zwsp_and_weekdays():
    html = (
        '<h3>суббота</h3>'
        '<h3>7\u00a0сентября</h3><p>a</p>'
        '<h3>10\u200b сентября</h3><p>b</p>'
        '<hr><p>nav</p>'
    )
    sections, rebuild = parse_month_sections(html)
    assert not rebuild
    assert [s.date for s in sections] == [date(2000, 9, 7), date(2000, 9, 10)]
    assert sections[0].start_idx == 1
    assert sections[0].end_idx == 3
    assert sections[1].start_idx == 3
    assert sections[1].end_idx == 5
