from __future__ import annotations


def test_normalize_event_type_board_game_meetup_maps_to_vstrecha() -> None:
    from main import normalize_event_type

    title = "🦄 Эволюция волшебных тварей"
    description = (
        'В арт-пространстве "Сигнал" пройдёт встреча для любителей настольных игр. '
        "Мастер игры объяснит правила перед партией. Количество игроков: 2–4 человека."
    )
    assert normalize_event_type(title, description, "мастер-класс") == "встреча"

