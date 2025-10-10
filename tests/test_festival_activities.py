from __future__ import annotations

from pathlib import Path

import pytest

from telegraph.utils import nodes_to_html

from festival_activities import (
    MAX_YAML_SIZE,
    FestivalActivitiesError,
    activities_to_telegraph_nodes,
    build_activity_card_lines,
    format_location_line,
    parse_festival_activities_yaml,
)


def test_parse_festival_activities_yaml_success():
    yaml_text = (
        "version: 2\n"
        "festival_site: https://fest.example\n"
        "always_on:\n"
        "  - title: Постоянная экспозиция\n"
        "    subtitle: Каждый день\n"
        "    time: 10:00-20:00\n"
        "    location:\n"
        "      name: Музей Янтаря\n"
        "    description: Описание\n"
        "    price: Бесплатно\n"
        "    cta:\n"
        "      url: https://fest.example/info\n"
        "by_request:\n"
        "  - title: Закрытая экскурсия\n"
        "    location: Неизвестное место\n"
    )

    result = parse_festival_activities_yaml(yaml_text)

    assert result.festival_site == "https://fest.example"
    assert len(result.groups) == 2
    always_on = result.groups[0]
    assert always_on.title == "Можно в любой день"
    card = always_on.items[0]
    assert card.location.name == "Музей Янтаря"
    assert card.cta_url == "https://fest.example/info"
    assert not card.cta_label or card.cta_label == "Подробнее"
    # Unknown location triggers a warning
    assert result.warnings
    assert "Неизвестное место" in result.warnings[0]

    payload = result.to_json_payload()
    assert payload[0]["kind"] == "meta"
    assert payload[1]["kind"] == "always_on"


def test_parse_festival_activities_yaml_size_limit():
    text = "version: 2\n" + "x" * (MAX_YAML_SIZE + 10)
    with pytest.raises(FestivalActivitiesError):
        parse_festival_activities_yaml(text)


def test_parse_festival_activities_yaml_requires_version():
    with pytest.raises(FestivalActivitiesError):
        parse_festival_activities_yaml("always_on: []\n")


def test_format_location_line():
    yaml_text = (
        "version: 2\n"
        "always_on:\n"
        "  - title: Доступно\n"
        "    location:\n"
        "      name: Музей Янтаря\n"
    )
    result = parse_festival_activities_yaml(yaml_text)
    location = result.groups[0].items[0].location
    assert format_location_line(location) == "Музей Янтаря, пл. Василевского 1, Калининград"


def test_activity_nodes_snapshot(tmp_path: Path):
    yaml_text = (
        "version: 2\n"
        "always_on:\n"
        "  - title: Экспозиция\n"
        "    subtitle: Каждый день\n"
        "    description: Добро пожаловать\n"
        "    cta:\n"
        "      label: Билеты\n"
        "      url: https://fest.example/tickets\n"
    )
    result = parse_festival_activities_yaml(yaml_text)
    nodes = activities_to_telegraph_nodes(result.groups)
    html = nodes_to_html(nodes)
    expected = (
        "<p>\u200b</p><p>\u200b</p><h3>Можно в любой день</h3>"
        "<h4>Экспозиция</h4><p>Каждый день</p><p>Добро пожаловать</p>"
        "<p><a href=\"https://fest.example/tickets\">Билеты</a></p><p>\u200b</p>"
    )
    assert html == expected


def test_build_activity_card_lines_order():
    yaml_text = (
        "version: 2\n"
        "always_on:\n"
        "  - title: Экспозиция\n"
        "    time: 10:00\n"
        "    price: Бесплатно\n"
        "    age: 12+\n"
        "    tags: [art, kids]\n"
    )
    result = parse_festival_activities_yaml(yaml_text)
    card = result.groups[0].items[0]
    lines = build_activity_card_lines(card)
    assert lines[0] == "🕒 10:00"
    assert lines[1] == "Бесплатно • 12+ • art, kids"
