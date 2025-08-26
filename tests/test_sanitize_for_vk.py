from markup import sanitize_for_vk


def test_sanitize_for_vk_strips_html_and_tg_emoji():
    src = (
        '<tg-emoji emoji-id="1"> 🗞 </tg-emoji> <i>Личный бренд на дзене</i>\n'
        'Приглашаем... <a href="https://forms.yandex.ru/u/68a392b2">регистрация</a><br>'
        '<i>Создаём своё имя...</i> <tg-spoiler>секрет</tg-spoiler>'
    )
    expected = (
        '🗞 _Личный бренд на дзене_\n'
        'Приглашаем... регистрация (https://forms.yandex.ru/u/68a392b2)\n'
        '_Создаём своё имя..._ секрет'
    )
    assert sanitize_for_vk(src) == expected

