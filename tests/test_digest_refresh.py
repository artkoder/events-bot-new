import pytest
from types import SimpleNamespace

import main


class BotStub:
    def __init__(self):
        self.deleted = []

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))


@pytest.mark.asyncio
async def test_handle_digest_refresh_masterclasses(monkeypatch):
    digest_id = "dg-test-master"
    session = {
        "chat_id": 1,
        "preview_msg_ids": [101],
        "panel_msg_id": 202,
        "items": [
            {
                "title": "Title 1",
                "norm_title": "Norm 1",
                "norm_description": "Desc 1",
                "event_type": "мастер-класс",
                "date": "2025-05-01",
                "end_date": None,
                "line_html": "<b>Item1</b>",
            },
            {
                "title": "Title 2",
                "norm_title": "Norm 2",
                "norm_description": "Desc 2",
                "event_type": "мастер-класс",
                "date": "2025-05-02",
                "end_date": None,
                "line_html": "<b>Item2</b>",
            },
        ],
        "intro_html": "Old intro",
        "footer_html": "Footer",
        "excluded": {1},
        "horizon_days": 7,
        "channels": [],
        "items_noun": "мастер-классов",
        "panel_text": "panel",
        "digest_type": "masterclasses",
    }
    main.digest_preview_sessions[digest_id] = session

    captured = {}

    async def fake_master(n, horizon, payload):
        captured["master_args"] = (n, horizon, payload)
        return "New intro"

    async def fail(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("unexpected composer call")

    async def fake_send_preview(session_obj, digest_id_arg, bot):
        captured["preview"] = (session_obj["intro_html"], digest_id_arg)
        return ("cap", False, 0, n)

    n = len(session["items"]) - len(session["excluded"])
    bot = BotStub()

    async def answer(**kwargs):
        captured["answered"] = True

    callback = SimpleNamespace(
        data=f"dg:r:{digest_id}",
        from_user=SimpleNamespace(id=1),
        message=None,
        answer=answer,
        id="cb1",
    )

    monkeypatch.setattr(main, "compose_masterclasses_intro_via_4o", fake_master)
    monkeypatch.setattr(main, "compose_exhibitions_intro_via_4o", fail)
    monkeypatch.setattr(main, "compose_digest_intro_via_4o", fail)
    monkeypatch.setattr(main, "_send_preview", fake_send_preview)

    try:
        await main.handle_digest_refresh(callback, bot)
    finally:
        main.digest_preview_sessions.pop(digest_id, None)

    assert captured["master_args"] == (
        n,
        session["horizon_days"],
        [{"title": "Norm 1", "description": "Desc 1"}],
    )
    assert captured["preview"] == ("New intro", digest_id)
    assert captured.get("answered")
    assert session["intro_html"] == "New intro"
    assert bot.deleted == [(1, 101), (1, 202)]


@pytest.mark.asyncio
async def test_handle_digest_refresh_exhibitions(monkeypatch):
    digest_id = "dg-test-exhib"
    session = {
        "chat_id": 2,
        "preview_msg_ids": [],
        "panel_msg_id": None,
        "items": [
            {
                "title": "Title 1",
                "norm_title": "Norm 1",
                "norm_description": "Desc 1",
                "event_type": "выставка",
                "date": "2025-06-01",
                "end_date": "2025-06-10",
                "line_html": "<b>Item1</b>",
            },
            {
                "title": "Title 2",
                "norm_title": "Norm 2",
                "norm_description": "Desc 2",
                "event_type": "выставка",
                "date": "2025-06-05",
                "end_date": None,
                "line_html": "<b>Item2</b>",
            },
        ],
        "intro_html": "Old intro",
        "footer_html": "Footer",
        "excluded": set(),
        "horizon_days": 14,
        "channels": [],
        "items_noun": "выставок",
        "panel_text": "panel",
        "digest_type": "exhibitions",
    }
    main.digest_preview_sessions[digest_id] = session

    captured = {}

    async def fake_exhib(n, horizon, payload):
        captured["exhib_args"] = (n, horizon, payload)
        return "Fresh intro"

    async def fail(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("unexpected composer call")

    async def fake_send_preview(session_obj, digest_id_arg, bot):
        captured["preview"] = (session_obj["intro_html"], digest_id_arg)
        return ("cap", False, 0, n)

    n = len(session["items"]) - len(session["excluded"])
    bot = BotStub()

    async def answer(**kwargs):
        captured["answered"] = True

    callback = SimpleNamespace(
        data=f"dg:r:{digest_id}",
        from_user=SimpleNamespace(id=1),
        message=None,
        answer=answer,
        id="cb2",
    )

    monkeypatch.setattr(main, "compose_masterclasses_intro_via_4o", fail)
    monkeypatch.setattr(main, "compose_exhibitions_intro_via_4o", fake_exhib)
    monkeypatch.setattr(main, "compose_digest_intro_via_4o", fail)
    monkeypatch.setattr(main, "_send_preview", fake_send_preview)

    try:
        await main.handle_digest_refresh(callback, bot)
    finally:
        main.digest_preview_sessions.pop(digest_id, None)

    assert captured["exhib_args"] == (
        n,
        session["horizon_days"],
        [
            {
                "title": "Norm 1",
                "description": "Desc 1",
                "date_range": {"start": "2025-06-01", "end": "2025-06-10"},
            },
            {
                "title": "Norm 2",
                "description": "Desc 2",
                "date_range": {"start": "2025-06-05", "end": "2025-06-05"},
            },
        ],
    )
    assert captured["preview"] == ("Fresh intro", digest_id)
    assert captured.get("answered")
    assert session["intro_html"] == "Fresh intro"
    assert bot.deleted == []


@pytest.mark.asyncio
async def test_handle_digest_refresh_psychology(monkeypatch):
    digest_id = "dg-test-psych"
    session = {
        "chat_id": 3,
        "preview_msg_ids": [501],
        "panel_msg_id": 601,
        "items": [
            {
                "title": "Mindfulness",
                "norm_title": "Mindfulness",
                "norm_description": "Calm",
                "event_type": "лекция",
                "date": "2025-06-02",
                "end_date": None,
                "line_html": "<b>Mind</b>",
                "norm_topics": ["PSYCHOLOGY", "wellbeing"],
            },
            {
                "title": "Other",
                "norm_title": "Other",
                "norm_description": "Desc",
                "event_type": "лекция",
                "date": "2025-06-05",
                "end_date": None,
                "line_html": "<b>Other</b>",
                "norm_topics": ["HISTORY"],
            },
        ],
        "intro_html": "Old intro",
        "footer_html": "Footer",
        "excluded": {1},
        "horizon_days": 7,
        "channels": [],
        "items_noun": "психологических событий",
        "panel_text": "panel",
        "digest_type": "psychology",
    }
    main.digest_preview_sessions[digest_id] = session

    captured: dict[str, object] = {}

    async def fake_psych(n, horizon, payload):
        captured["psych_args"] = (n, horizon, payload)
        return "New psych intro"

    async def fail(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("unexpected composer call")

    async def fake_send_preview(session_obj, digest_id_arg, bot):
        captured["preview"] = (session_obj["intro_html"], digest_id_arg)
        return ("cap", False, 0, n)

    n = len(session["items"]) - len(session["excluded"])
    bot = BotStub()

    async def answer(**kwargs):
        captured["answered"] = True

    callback = SimpleNamespace(
        data=f"dg:r:{digest_id}",
        from_user=SimpleNamespace(id=1),
        message=None,
        answer=answer,
        id="cb3",
    )

    monkeypatch.setattr(main, "compose_masterclasses_intro_via_4o", fail)
    monkeypatch.setattr(main, "compose_exhibitions_intro_via_4o", fail)
    monkeypatch.setattr(main, "compose_digest_intro_via_4o", fail)
    monkeypatch.setattr(main, "compose_psychology_intro_via_4o", fake_psych)
    monkeypatch.setattr(main, "_send_preview", fake_send_preview)

    try:
        await main.handle_digest_refresh(callback, bot)
    finally:
        main.digest_preview_sessions.pop(digest_id, None)

    assert captured["psych_args"] == (
        n,
        session["horizon_days"],
        [
            {
                "title": "Mindfulness",
                "description": "Calm",
                "topics": ["PSYCHOLOGY", "wellbeing"],
            }
        ],
    )
    assert captured["preview"] == ("New psych intro", digest_id)
    assert captured.get("answered")
    assert session["intro_html"] == "New psych intro"
    assert bot.deleted == [(3, 501), (3, 601)]
