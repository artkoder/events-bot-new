import json
from pathlib import Path

import pytest
from aiogram import types
from db import Database
import main
from main_part2 import event_to_nodes
from models import Event, User
from preview_3d.handlers import (
    _build_month_batch_menu,
    _build_preview3d_runtime_config_payload,
    _build_preview3d_runtime_secrets_payload,
    _prepare_preview3d_runtime_datasets,
    _push_preview3d_kernel,
    handle_3di_command,
    handle_3di_callback,
    update_previews_from_results,
)
from source_parsing.telegram.split_secrets import decrypt_secret


DENIED_TEXT = "\u274c \u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043f\u0440\u0430\u0432"


class DummyBot:
    def __init__(self):
        self.messages = []
        self.edits = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": kwargs.get("reply_markup"),
            }
        )

    async def edit_message_text(self, *, chat_id, message_id, text, **kwargs):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": kwargs.get("reply_markup"),
            }
        )


class DummyCallback:
    def __init__(self, data: str, user_id: int = 1, chat_id: int = 1, message_id: int = 1):
        self.data = data
        self.from_user = type("FromUser", (), {"id": user_id})()
        self.message = type(
            "Message",
            (),
            {
                "chat": type("Chat", (), {"id": chat_id})(),
                "message_id": message_id,
            },
        )()
        self.answers = []

    async def answer(self, text=None, show_alert=False):
        self.answers.append({"text": text, "show_alert": show_alert})


def _make_event(event_id: int, **overrides: object) -> Event:
    payload = {
        "id": event_id,
        "title": f"Event {event_id}",
        "description": "Desc",
        "date": "2026-05-15",
        "time": "19:00",
        "location_name": "Loc",
        "source_text": "src",
    }
    payload.update(overrides)
    return Event(**payload)


def test_event_to_nodes_prefers_preview_3d_url():
    event = _make_event(
        1,
        photo_urls=["https://example.com/photo.jpg"],
        preview_3d_url="https://example.com/preview.jpg",
    )
    nodes = event_to_nodes(event, show_image=True)
    assert nodes[0]["tag"] == "figure"
    assert nodes[0]["children"][0]["attrs"]["src"] == "https://example.com/preview.jpg"


def test_event_to_nodes_falls_back_to_photo_urls():
    event = _make_event(
        1,
        photo_urls=["https://example.com/photo.jpg"],
        preview_3d_url="not-a-url",
    )
    nodes = event_to_nodes(event, show_image=True)
    assert nodes[0]["tag"] == "figure"
    assert nodes[0]["children"][0]["attrs"]["src"] == "https://example.com/photo.jpg"


def test_build_month_batch_menu_offers_50_and_all():
    markup = _build_month_batch_menu("2026-04", 55)
    buttons = [button for row in markup.inline_keyboard for button in row]

    assert [button.text for button in buttons] == [
        "Первые 25",
        "Первые 50",
        "Все (55)",
        "⬅️ К месяцам",
    ]
    assert buttons[1].callback_data == "3di:genbatch:2026-04:50"


@pytest.mark.asyncio
async def test_build_source_page_content_swaps_webp_cover_by_default():
    html, _, _ = await main.build_source_page_content(
        "Title",
        "Body",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[
            "https://example.com/preview.webp",
            "https://example.com/other.jpg",
        ],
    )
    assert html.startswith('<figure><img src="https://example.com/other.jpg"/>')


@pytest.mark.asyncio
async def test_build_source_page_content_force_cover_keeps_webp():
    html, _, _ = await main.build_source_page_content(
        "Title",
        "Body",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[
            "https://example.com/preview.webp",
            "https://example.com/other.jpg",
        ],
        force_cover_url="https://example.com/preview.webp",
    )
    assert html.startswith('<figure><img src="https://example.com/preview.webp"/>')


@pytest.mark.asyncio
async def test_3di_command_denied_without_superadmin(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=False))
        await session.commit()

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "/3di",
        }
    )
    bot = DummyBot()
    await handle_3di_command(msg, db, bot)

    assert bot.messages
    assert bot.messages[0]["text"] == DENIED_TEXT


@pytest.mark.asyncio
async def test_update_previews_from_results_updates_db(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        session.add(_make_event(2))
        await session.commit()

    results = [
        {
            "event_id": 1,
            "preview_url": "https://example.com/preview.jpg",
            "status": "ok",
        },
        {
            "event_id": 2,
            "preview_url": "https://example.com/other.jpg",
            "status": "error",
            "error": "render failed",
        },
    ]
    updated, errors, skipped = await update_previews_from_results(db, results)

    assert updated == 1
    assert errors == 1
    assert skipped == 0

    async with db.get_session() as session:
        ev1 = await session.get(Event, 1)
        ev2 = await session.get(Event, 2)

    assert ev1.preview_3d_url == "https://example.com/preview.jpg"
    assert ev2.preview_3d_url is None


@pytest.mark.asyncio
async def test_update_previews_from_results_handles_skip(tmp_path):
    """Test that skip status is counted separately from errors."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(_make_event(1))
        await session.commit()

    results = [
        {"event_id": 1, "status": "skip", "error": "No images"},
    ]
    updated, errors, skipped = await update_previews_from_results(db, results)

    assert updated == 0
    assert errors == 0
    assert skipped == 1


@pytest.mark.asyncio
async def test_get_new_events_gap(tmp_path):
    """Test that _get_new_events_gap returns events after the last one with preview."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    from preview_3d.handlers import _get_new_events_gap

    # Create 4 events:
    # 4 (Newest) - No preview, 2 images -> Should return
    # 3          - No preview, 0 images -> Should skip (if min_images=1)
    # 2          - Has preview          -> Stop barrier
    # 1 (Oldest) - No preview           -> Should not reach
    
    async with db.get_session() as session:
        session.add(_make_event(1, photo_urls=["http://img"], preview_3d_url=None))
        session.add(_make_event(2, photo_urls=["http://img"], preview_3d_url="http://preview"))
        session.add(_make_event(3, photo_urls=[], preview_3d_url=None))
        session.add(_make_event(4, photo_urls=["http://img"], preview_3d_url=None))
        await session.commit()
        
    candidates = await _get_new_events_gap(db, min_images=1)
    
    # Updated logic: does NOT stop at barrier.
    # Should get 4 (newest) AND 1 (oldest, behind barrier w/ proper images).
    # Event 3 skipped due to no images.
    
    assert len(candidates) == 2
    ids = sorted([e.id for e in candidates])
    assert ids == [1, 4]
    
    # Test with min_images=0, should get 4, 3, 1
    candidates_all = await _get_new_events_gap(db, min_images=0)
    assert len(candidates_all) == 3
    ids = sorted([e.id for e in candidates_all])
    assert ids == [1, 3, 4]


@pytest.mark.asyncio
async def test_handle_3di_month_selection_opens_batch_menu(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        session.add(_make_event(1, date="2026-04-10", photo_urls=["http://img-1"], preview_3d_url=None))
        session.add(
            _make_event(
                2,
                date="2026-04-11",
                photo_urls=["http://img-2"],
                preview_3d_url="https://example.com/existing.webp",
            )
        )
        session.add(_make_event(3, date="2026-04-12", photo_urls=["http://img-3"], preview_3d_url=None))
        await session.commit()

    callback = DummyCallback("3di:gen:2026-04", user_id=1)
    bot = DummyBot()

    await handle_3di_callback(callback, db, bot)

    assert len(bot.edits) == 1
    assert "Без 3D-превью: 2" in bot.edits[0]["text"]
    markup = bot.edits[0]["reply_markup"]
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert [button.text for button in buttons] == ["Все (2)", "⬅️ К месяцам"]
    assert buttons[0].callback_data == "3di:genbatch:2026-04:all"
    assert callback.answers == [{"text": None, "show_alert": False}]


@pytest.mark.asyncio
async def test_handle_3di_month_generation_uses_only_missing_previews_and_batch_limit(
    tmp_path, monkeypatch
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        for idx in range(1, 56):
            day = ((idx - 1) % 28) + 1
            hour = 10 + ((idx - 1) // 28)
            session.add(
                _make_event(
                    idx,
                    date=f"2026-04-{day:02d}",
                    time=f"{hour:02d}:00",
                    photo_urls=[f"http://img-{idx}"],
                    preview_3d_url=None,
                )
            )
        session.add(
            _make_event(
                999,
                date="2026-04-29",
                time="18:00",
                photo_urls=["http://img-999"],
                preview_3d_url="https://example.com/existing.webp",
            )
        )
        await session.commit()

    captured = {}

    async def fake_start_generation(
        db_arg,
        bot_arg,
        callback_arg,
        events,
        month,
        mode,
        start_kaggle_render,
        operator_id,
        total_event_count,
        batch_limit,
    ):
        captured["event_ids"] = [event.id for event in events]
        captured["month"] = month
        captured["mode"] = mode
        captured["operator_id"] = operator_id
        captured["total_event_count"] = total_event_count
        captured["batch_limit"] = batch_limit

    monkeypatch.setattr("preview_3d.handlers._start_generation", fake_start_generation)

    callback = DummyCallback("3di:genbatch:2026-04:50", user_id=1)
    bot = DummyBot()

    await handle_3di_callback(callback, db, bot)

    assert captured["month"] == "2026-04"
    assert captured["mode"] == "month"
    assert captured["operator_id"] == 1
    assert captured["total_event_count"] == 55
    assert captured["batch_limit"] == 50
    assert len(captured["event_ids"]) == 50
    assert 999 not in captured["event_ids"]
    assert callback.answers == []


def test_build_preview3d_runtime_payloads(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_KEY", "service-key")
    monkeypatch.setenv("SUPABASE_MEDIA_BUCKET", "events-media")
    monkeypatch.setenv("SUPABASE_PREVIEW3D_PREFIX", "p3d-custom")

    config = _build_preview3d_runtime_config_payload()
    secrets = json.loads(_build_preview3d_runtime_secrets_payload())

    assert config["env"]["SUPABASE_BUCKET"] == "events-media"
    assert config["env"]["SUPABASE_MEDIA_BUCKET"] == "events-media"
    assert config["env"]["SUPABASE_PREVIEW3D_PREFIX"] == "p3d-custom"
    assert secrets == {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "service-key",
        "SUPABASE_SERVICE_KEY": "service-key",
    }


@pytest.mark.asyncio
async def test_prepare_preview3d_runtime_datasets_writes_config_and_encrypted_secrets(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")
    monkeypatch.setenv("SUPABASE_MEDIA_BUCKET", "events-media")
    monkeypatch.setenv("SUPABASE_PREVIEW3D_PREFIX", "p3d-custom")
    monkeypatch.setenv("KAGGLE_USERNAME", "zigomaro")

    captured: dict[str, dict[str, bytes | str]] = {}

    def fake_create_dataset(client, username, slug_suffix, title, writer):
        dataset_dir = tmp_path / slug_suffix
        dataset_dir.mkdir(parents=True, exist_ok=True)
        writer(dataset_dir)
        files: dict[str, bytes | str] = {}
        for child in dataset_dir.iterdir():
            if child.suffix in {".json"}:
                files[child.name] = child.read_text(encoding="utf-8")
            else:
                files[child.name] = child.read_bytes()
        captured[slug_suffix] = files
        return f"{username}/{slug_suffix}"

    monkeypatch.setattr("preview_3d.handlers._create_dataset", fake_create_dataset)

    slug_cipher, slug_key = await _prepare_preview3d_runtime_datasets(
        client=object(),
        run_id="run-123",
    )

    assert slug_cipher.startswith("zigomaro/preview3d-runtime-cipher")
    assert slug_key.startswith("zigomaro/preview3d-runtime-key")

    cipher_files = captured[Path(slug_cipher).name]
    key_files = captured[Path(slug_key).name]
    assert "fernet.key" not in cipher_files
    assert list(key_files) == ["fernet.key"]
    config = json.loads(str(cipher_files["config.json"]))
    decrypted = decrypt_secret(
        cipher_files["secrets.enc"],
        key_files["fernet.key"],
    )

    assert config["env"]["SUPABASE_BUCKET"] == "events-media"
    assert config["env"]["SUPABASE_MEDIA_BUCKET"] == "events-media"
    assert config["env"]["SUPABASE_PREVIEW3D_PREFIX"] == "p3d-custom"
    assert json.loads(decrypted) == {
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_KEY": "service-key",
        "SUPABASE_SERVICE_KEY": "service-key",
    }


@pytest.mark.asyncio
async def test_push_preview3d_kernel_uses_shared_kaggle_client(monkeypatch, tmp_path):
    kernel_dir = tmp_path / "Preview3D"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "zigomaro/preview-3d",
                "slug": "preview-3d",
                "title": "Preview 3D",
                "dataset_sources": [],
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    class DummyClient:
        def push_kernel(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("preview_3d.handlers.KERNELS_ROOT_PATH", tmp_path)
    monkeypatch.setattr("preview_3d.handlers.KAGGLE_STARTUP_WAIT_SECONDS", 0)
    monkeypatch.setenv("KAGGLE_USERNAME", "zigomaro")

    kernel_ref = await _push_preview3d_kernel(
        DummyClient(),
        ["zigomaro/cipher", "zigomaro/key"],
    )

    assert kernel_ref == "zigomaro/preview-3d"
    assert calls == [
        {
            "kernel_path": kernel_dir,
            "dataset_sources": ["zigomaro/cipher", "zigomaro/key"],
        }
    ]
