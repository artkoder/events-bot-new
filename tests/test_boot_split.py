import asyncio
import builtins
import json
import os
import py_compile
import sys
import textwrap
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiogram import types

import vk_review


START_MARKER = "# --- split-loader"
END_MARKER = "# --- end split-loader ---"


def _extract_loader_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    recording = False
    for line in text.splitlines():
        if line.startswith(START_MARKER):
            recording = True
            continue
        if line.startswith(END_MARKER):
            break
        if recording:
            lines.append(line)
    loader_src = "\n".join(lines)
    if not loader_src.strip():
        raise AssertionError("split-loader block not found in main.py")
    return loader_src


async def _collect_handler_counts(env: dict[str, str], cwd: Path) -> dict[str, int]:
    script = textwrap.dedent(
        """
        import json
        import os
        import sys

        sys.path.insert(0, os.getcwd())

        import aiogram.webhook.aiohttp_server as server
        import main

        captured = {}
        original = server.SimpleRequestHandler.register

        def capture(self, app, *args, **kwargs):
            captured['dispatcher'] = self.dispatcher
            return original(self, app, *args, **kwargs)

        server.SimpleRequestHandler.register = capture
        try:
            main.create_app()
        finally:
            server.SimpleRequestHandler.register = original

        dp = captured['dispatcher']
        print(json.dumps({
            "message": len(dp.message.handlers),
            "callback": len(dp.callback_query.handlers),
            "chat_member": len(dp.my_chat_member.handlers),
        }))
        """
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            f"handler count script failed with {proc.returncode}: {stderr.decode()}"
        )
    output = stdout.decode().strip().splitlines()
    if not output:
        raise AssertionError("handler count script produced no output")
    return json.loads(output[-1])


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[SimpleNamespace] = []

    async def send_message(self, chat_id: int, text: str, **kwargs) -> SimpleNamespace:
        message = SimpleNamespace(
            message_id=len(self.messages) + 1,
            date=0,
            chat=SimpleNamespace(id=chat_id, type="private"),
            text=text,
            reply_markup=kwargs.get("reply_markup"),
        )
        self.messages.append(message)
        return message


def test_split_loader_propagates_syntax_error(monkeypatch):
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    loader_block = _extract_loader_block(main_path)

    original_compile = builtins.compile

    def failing_compile(source, filename, mode, *args, **kwargs):
        if filename == "main_part2.py":
            raise SyntaxError("boom")
        return original_compile(source, filename, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", failing_compile)

    module_globals = {
        "__file__": str(main_path),
        "__builtins__": builtins,
    }

    with pytest.raises(SyntaxError):
        exec(loader_block, module_globals, module_globals)


@pytest.mark.usefixtures("no_network")
@pytest.mark.asyncio
async def test_boot_split_smoke(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    main_path = repo_root / "main.py"
    main_part2_path = repo_root / "main_part2.py"

    # Sanity checks to catch accidental regressions early.
    for path in (main_path, main_part2_path):
        assert path.stat().st_size < 900 * 1024, f"{path.name} grew past 900KB"
        py_compile.compile(str(path), doraise=True)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "WEBHOOK_URL": "https://example.invalid/webhook",
            "DB_PATH": str(tmp_path / "boot.sqlite"),
        }
    )

    # Import main in a fresh interpreter to ensure the split-loader succeeds.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import main",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(repo_root),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess import failed: {stdout.decode()}\n{stderr.decode()}"
        )

    # Launch the script to verify the aiohttp bootstrap doesn't raise immediately.
    main_proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "main.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(repo_root),
    )
    try:
        await asyncio.sleep(2)
        assert main_proc.returncode is None, "main.py exited prematurely"
    finally:
        main_proc.terminate()
        await asyncio.wait_for(main_proc.communicate(), timeout=30)

    counts_once = await _collect_handler_counts(env, repo_root)
    counts_again = await _collect_handler_counts(env, repo_root)
    assert counts_once == counts_again

    # Manual command invocations within the current interpreter.
    import main

    db_file = tmp_path / "state.sqlite"
    database = main.Database(str(db_file))
    await database.init()

    async with database.get_session() as session:
        session.add(main.User(user_id=1, is_superadmin=True))
        session.add(
            main.Event(
                title="Calendar Showcase",
                description="Demo description",
                date=date.today().isoformat(),
                time="18:00",
                location_name="Venue",
                city="TestCity",
                telegraph_url="https://t.me/example",
                ics_url="https://example.invalid/demo.ics",
                source_text="Original source",
            )
        )
        await session.commit()

    bot = DummyBot()

    help_message = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Tester"},
            "text": "/help",
        }
    )
    await main.handle_help(help_message, database, bot)
    assert bot.messages[-1].text.strip(), "help command produced empty response"

    events_message = types.Message.model_validate(
        {
            "message_id": 2,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Tester"},
            "text": "/events",
        }
    )
    await main.handle_events(events_message, database, bot)
    calendar_text = bot.messages[-1].text
    assert "Calendar Showcase" in calendar_text

    async def fake_build_source_page_content(*args, **kwargs):
        return "<p>content</p>", "", 0

    class FakeTelegraph:
        def __init__(self, access_token: str) -> None:
            self.token = access_token

    monkeypatch.setattr(main, "build_source_page_content", fake_build_source_page_content)
    monkeypatch.setattr(main, "Telegraph", FakeTelegraph)

    telegraph_result = await main.create_source_page(
        "Title",
        "Body",
        "https://example.invalid/source",
        ics_url="https://example.invalid/cal.ics",
        db=database,
    )
    assert telegraph_result and telegraph_result[0], "Telegraph routine returned empty data"

    monkeypatch.setattr(main, "VK_AFISHA_GROUP_ID", -777)
    monkeypatch.setattr(main, "VK_PHOTOS_ENABLED", False)

    async def fake_ensure_vk_short_ics_link(*args, **kwargs):
        return ("https://vk.cc/demo", "demo")

    async def fake_ensure_vk_short_ticket_link(*args, **kwargs):
        return ("https://vk.cc/ticket", "ticket")

    async def fake_post_to_vk(*args, **kwargs):
        return "https://vk.com/wall-777_1"

    def fake_build_vk_source_message(event, text, festival=None, calendar_url=None):
        return f"{text} :: {calendar_url}"

    monkeypatch.setattr(main, "ensure_vk_short_ics_link", fake_ensure_vk_short_ics_link)
    monkeypatch.setattr(main, "ensure_vk_short_ticket_link", fake_ensure_vk_short_ticket_link)
    monkeypatch.setattr(main, "post_to_vk", fake_post_to_vk)
    monkeypatch.setattr(main, "build_vk_source_message", fake_build_vk_source_message)

    vk_event = main.Event(
        title="VK Routine",
        description="VK body",
        date=date.today().isoformat(),
        time="12:00",
        location_name="Venue",
        source_text="VK source",
    )
    vk_url = await main.sync_vk_source_post(
        vk_event,
        "VK payload",
        database,
        None,
        ics_url="https://example.invalid/vk.ics",
    )
    assert vk_url == "https://vk.com/wall-777_1"
    assert vk_event.ics_url == "https://example.invalid/vk.ics"

    async def fake_get_tz_offset(_db):
        return "+00:00"

    monkeypatch.setattr(vk_review, "require_main_attr", lambda name: fake_get_tz_offset)
    monkeypatch.setattr(
        vk_review,
        "extract_event_ts_hint",
        lambda text, publish_ts=None, allow_past=False: 1_700_003_600,
    )

    async with database.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox (group_id, post_id, date, text, matched_kw, has_date, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, 1, 1_700_000_000, "Demo inbox entry", "kw", 0, "pending"),
        )
        await conn.commit()

    updated = await vk_review.refresh_vk_event_ts_hints(database)
    assert updated >= 1

    async with database.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT event_ts_hint FROM vk_inbox WHERE group_id=? AND post_id=?",
            (1, 1),
        )
        row = await cursor.fetchone()
    assert row and row[0] == 1_700_003_600
