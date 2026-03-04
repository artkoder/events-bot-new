from event_utils import strip_city_from_address


def test_strip_city_from_address_compacts_ul_prefix_and_commas() -> None:
    assert (
        strip_city_from_address("ул. Тельмана, 28, Калининград", "Калининград")
        == "Тельмана 28"
    )

