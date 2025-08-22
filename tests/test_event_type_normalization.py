from main import normalize_event_type


def test_normalize_event_type_film():
    title = "\ud83c\udfa5 Очень страшное кино"
    desc = "Показ культового фильма 'Очень страшное кино' в Заре."
    assert normalize_event_type(title, desc, "спектакль") == "кинопоказ"


def test_normalize_event_type_no_change():
    title = "\ud83c\udfb5 Concert"
    desc = "Музыкальный концерт"
    assert normalize_event_type(title, desc, "концерт") == "концерт"


def test_normalize_event_type_masterclass():
    title = "\ud83d\udcda Workshop"
    desc = "Практический мастер-класс"
    assert normalize_event_type(title, desc, "мастер-класс") == "мастер-класс"
