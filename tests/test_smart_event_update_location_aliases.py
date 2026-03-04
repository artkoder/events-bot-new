from __future__ import annotations

import smart_event_update as su


def test_scientific_library_aliases_normalize_to_one_location() -> None:
    assert su._normalize_location("Научная библиотека") == "научная библиотека"
    assert (
        su._normalize_location("Научная библиотека, Мира 9, Калининград")
        == "научная библиотека"
    )
    assert (
        su._normalize_location("Калининградская областная научная библиотека")
        == "научная библиотека"
    )


def test_scientific_library_bfu_is_not_treated_as_same_alias() -> None:
    assert su._normalize_location("Научная библиотека БФУ") != "научная библиотека"


def test_allow_parallel_events_for_scientific_library_aliases() -> None:
    su._load_location_flags.cache_clear()
    assert su._allow_parallel_events("Научная библиотека")
    assert su._allow_parallel_events("Научная библиотека, Мира 9, Калининград")
    assert su._allow_parallel_events("Калининградская областная научная библиотека")
    assert not su._allow_parallel_events("Научная библиотека БФУ")


def test_dom_kitoboya_aliases_normalize_to_one_location() -> None:
    assert su._normalize_location("Дом китобоя") == "дом китобоя"
    assert su._normalize_location("Дом китобоя, пр-т Мира 9, Калининград") == "дом китобоя"


def test_zakheim_aliases_normalize_to_one_location() -> None:
    assert su._normalize_location("Закхаймские ворота") == "закхаймские ворота"
    assert su._normalize_location("Арт-пространство Ворота") == "закхаймские ворота"


def test_location_noise_prefixes_do_not_create_new_locations() -> None:
    assert su._normalize_location("Кинотеатр Сигнал") == "сигнал"
    assert su._normalize_location("Арт-пространство Сигнал") == "сигнал"
