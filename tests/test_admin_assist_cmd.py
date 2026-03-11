import os
import pathlib
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

import main  # noqa: F401  # ensure "main" module is loaded for require_main_attr()
from handlers import admin_assist_cmd as assist


def _registered_commands() -> set[str]:
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [
        root / "main_part2.py",
        root / "handlers" / "telegraph_cache_cmd.py",
        root / "handlers" / "special_cmd.py",
        root / "handlers" / "recent_imports_cmd.py",
        root / "handlers" / "popular_posts_cmd.py",
        root / "handlers" / "ik_poster_cmd.py",
        root / "handlers" / "admin_assist_cmd.py",
        root / "source_parsing" / "commands.py",
        root / "source_parsing" / "telegram" / "commands.py",
    ]
    commands: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        commands.update(re.findall(r'Command\("([A-Za-z0-9_]+)"\)', text))
    commands.add(main.VK_MISS_REVIEW_COMMAND.lstrip("/"))
    return commands


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
    assert assist._build_command_text("recent_imports", {"args_text": "48"}) == "/recent_imports 48"
    with pytest.raises(ValueError):
        assist._build_command_text("parse", {"args_text": "check\nrm -rf"})
    with pytest.raises(ValueError):
        assist._build_command_text("parse", {"args_text": "/help"})


def test_build_command_trace_requires_arg():
    with pytest.raises(ValueError):
        assist._build_command_text("trace", {})


def test_build_command_rebuild_event_requires_arg():
    assert assist._build_command_text("rebuild_event", {"args_text": "123 --regen-desc"}) == "/rebuild_event 123 --regen-desc"
    with pytest.raises(ValueError):
        assist._build_command_text("rebuild_event", {})


def test_direct_command_proposal_preserves_explicit_command_and_args():
    proposal = assist._extract_direct_command_proposal("запусти /rebuild_event 321 --regen-desc")
    assert proposal is not None
    assert proposal.action_id == "rebuild_event"
    assert proposal.args == {"args_text": "321 --regen-desc"}


def test_direct_command_proposal_for_simple_noarg_command():
    proposal = assist._extract_direct_command_proposal("команда recent_imports")
    assert proposal is not None
    assert proposal.action_id == "recent_imports"
    assert proposal.args == {}


def test_allowlist_covers_registered_commands():
    commands = _registered_commands()
    allow = set(assist._allowed_actions().keys())
    missing = commands - allow
    assert missing == set()


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


def test_heuristic_routes_recent_imports_legacy_phrase():
    props = assist._heuristic_proposals(
        "какие события были созданы или обновлены из телеграма, вк и parse за последние 24 часа"
    )
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "recent_imports"


def test_heuristic_routes_recent_imports_for_source_list_query():
    props = assist._heuristic_proposals("список событий созданных из телеграм и вк за сутки")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "recent_imports"
    assert props[0].args == {}


def test_heuristic_routes_recent_imports_with_explicit_hours():
    props = assist._heuristic_proposals("покажи свежие импортированные события за 48 часов")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "recent_imports"
    assert props[0].args == {"args_text": "48"}


def test_heuristic_routes_telegraph_cache_stats():
    props = assist._heuristic_proposals("покажи статистику кэша телеграф для событий")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "telegraph_cache_stats"
    assert props[0].args == {"args_text": "event"}


def test_heuristic_routes_telegraph_cache_sanitize():
    props = assist._heuristic_proposals("прогрей кэш телеграф страниц")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "telegraph_cache_sanitize"


def test_heuristic_routes_ik_poster():
    props = assist._heuristic_proposals("обработай афишу через imagekit")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "ik_poster"


def test_heuristic_keeps_count_queries_on_general_stats():
    props = assist._heuristic_proposals("сколько событий создал бот из автоимпорта за сутки")
    assert props, "expected heuristic proposals"
    assert props[0].action_id == "general_stats"
