import main


def test_sanitize_telegram_html_variants():
    f = main.sanitize_telegram_html
    assert f("<tg-emoji a='1'/>") == ""
    assert f("<tg-emoji a='1'></tg-emoji>") == ""
    assert f("<tg-emoji a='1'>➡</tg-emoji>") == "➡"
    assert f("&lt;tg-emoji a='1'/&gt;") == ""
    assert f("&lt;tg-emoji a='1'&gt;&lt;/tg-emoji&gt;") == ""
    assert f("&lt;tg-emoji a='1'&gt;➡&lt;/tg-emoji&gt;") == "➡"
