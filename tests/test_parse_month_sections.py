from datetime import date

from sections import parse_month_sections


def test_parse_month_sections_basic():
    html = (
        '<h3>🟥🟥🟥 9 сентября 🟥🟥🟥</h3>'
        '<p>\u200b</p><h4>A</h4><p>\u200b</p>'
        '<h3>🟥🟥🟥 10 сентября 🟥🟥🟥</h3><p>\u200b</p>'
    )
    sections = parse_month_sections(html)
    assert [s.date for s in sections] == [date(2000, 9, 9), date(2000, 9, 10)]
    assert sections[0].h3_idx == 0
    assert sections[0].start_idx == 1
    assert sections[0].end_idx == 4
    assert sections[1].start_idx == 5


def test_parse_month_sections_spaces_and_case():
    html = (
        '<h3>🟥🟥🟥  9  СЕНТЯБРЯ  🟥🟥🟥</h3>'
        '<p>\u200b</p>'
    )
    sections = parse_month_sections(html)
    assert len(sections) == 1
    assert sections[0].date == date(2000, 9, 9)
