import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

import main  # noqa: F401  # ensure "main" module is loaded for require_main_attr()
from handlers import admin_assist_cmd as assist


def test_build_command_events_date_optional():
    assert assist._build_command_text("events", {"date": "2026-02-24"}) == "/events 2026-02-24"
    assert assist._build_command_text("events", {}) == "/events"


def test_build_command_tz_offset():
    assert assist._build_command_text("tz", {"offset": "+02:00"}) == "/tz +02:00"
    with pytest.raises(ValueError):
        assist._build_command_text("tz", {"offset": "2"})


def test_build_command_vkgroup_value():
    assert assist._build_command_text("vkgroup", {"value": "off"}) == "/vkgroup off"
    assert assist._build_command_text("vkgroup", {"value": "12345"}) == "/vkgroup 12345"
    with pytest.raises(ValueError):
        assist._build_command_text("vkgroup", {"value": "abc"})


def test_build_command_vktime():
    assert assist._build_command_text("vktime", {"kind": "today", "time": "08:00"}) == "/vktime today 08:00"
    with pytest.raises(ValueError):
        assist._build_command_text("vktime", {"kind": "tomorrow", "time": "08:00"})


def test_build_command_args_text_sanitized():
    assert assist._build_command_text("parse", {"args_text": "check"}) == "/parse check"
    with pytest.raises(ValueError):
        assist._build_command_text("parse", {"args_text": "check\nrm -rf"})
    with pytest.raises(ValueError):
        assist._build_command_text("parse", {"args_text": "/help"})


def test_build_command_trace_requires_arg():
    with pytest.raises(ValueError):
        assist._build_command_text("trace", {})


def test_validate_plan_proposals_ok():
    kind, proposals, question, options = assist._validate_plan(
        {
            "kind": "proposals",
            "proposals": [
                {
                    "action_id": "events",
                    "args": {"date": "2026-02-24"},
                    "confidence": 0.9,
                }
            ],
        }
    )
    assert kind == "proposals"
    assert proposals and proposals[0].action_id == "events"
    assert question == ""
    assert options == []


def test_validate_plan_clarify_ok():
    kind, proposals, question, options = assist._validate_plan(
        {
            "kind": "clarify",
            "question": "Какой день?",
            "options": [
                {"label": "Сегодня", "add_to_request": "today"},
                {"label": "Завтра", "add_to_request": "tomorrow"},
            ],
        }
    )
    assert kind == "clarify"
    assert proposals == []
    assert question == "Какой день?"
    assert len(options) == 2


def test_validate_plan_unknown_when_no_valid_actions():
    kind, proposals, question, options = assist._validate_plan(
        {
            "kind": "proposals",
            "proposals": [
                {"action_id": "does_not_exist", "args": {}},
            ],
        }
    )
    assert kind == "unknown"
    assert proposals == []
    assert question == ""
    assert options == []


def test_heuristic_routes_general_stats_for_system_phrase():
    props = assist._heuristic_proposals("общая статистика по авторазбору вк и телеграм и гемма")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "general_stats"


def test_heuristic_routes_popular_posts():
    props = assist._heuristic_proposals("посмотреть статистику популярных постов")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "popular_posts"
