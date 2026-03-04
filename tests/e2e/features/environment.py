"""
Behave environment hooks for E2E Telegram bot testing.

Initializes HumanUserClient before test suite and cleans up after.
"""

import asyncio
import os
import sys
import logging
import json
import urllib.request
import sqlite3
import subprocess
import time
import contextlib
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.e2e.human_client import HumanUserClient, create_human_client

logger = logging.getLogger("e2e.behave")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def _is_offline_mode() -> bool:
    return (os.getenv("E2E_OFFLINE") or "").strip() in {"1", "true", "yes"}

def _is_manual_run_enabled() -> bool:
    """Allow heavy @manual scenarios only when explicitly requested.

    There are two explicit opt-ins:
    - E2E_RUN_MANUAL=1 (recommended for CI-like runs)
    - behave invoked with a manual tag filter (e.g. --tags=manual)
    """
    if (os.getenv("E2E_RUN_MANUAL") or "").strip() == "1":
        return True

    # Best-effort: detect explicit tag filter in behave argv.
    # Examples:
    # - behave ... --tags=manual
    # - behave ... --tags manual
    # - behave ... -t manual
    raw: list[str] = []
    argv = list(sys.argv or [])
    for i, arg in enumerate(argv):
        if arg.startswith("--tags="):
            raw.append(arg.split("=", 1)[1])
            continue
        if arg in {"--tags", "-t"} and (i + 1) < len(argv):
            raw.append(argv[i + 1])
            continue

    if not raw:
        return False

    haystack = " ".join(raw).lower().replace("@", " ")
    return "manual" in {x.strip() for x in haystack.replace(",", " ").split() if x.strip()}

def _ensure_db_path_env() -> None:
    """Set DB_PATH for local E2E runs when it's missing.

    The bot defaults DB_PATH to /data/db.sqlite, but E2E steps fall back to a
    prod snapshot sqlite in repo root. If DB_PATH is not set, tests and bot end
    up using different DB files, which makes Telegram Monitoring look broken
    (sources=0, no events).
    """
    if (os.getenv("DB_PATH") or "").strip():
        return

    def _sqlite_snapshot_ok(path: Path) -> bool:
        try:
            conn = sqlite3.connect(str(path))
            try:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            finally:
                conn.close()
            return True
        except Exception:
            return False

    base = project_root / "db_prod_snapshot.sqlite"
    globbed = sorted(
        [p for p in project_root.glob("db_prod_snapshot_*.sqlite") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer the conventional "current" snapshot if it's healthy; otherwise fall back
    # to the newest timestamped snapshot available.
    candidates: list[Path] = ([base] if base.exists() else []) + globbed

    for p in candidates:
        if not p.exists():
            continue
        if _sqlite_snapshot_ok(p):
            os.environ["DB_PATH"] = str(p)
            return
        logger.warning("E2E DB snapshot is malformed (skipping): %s", p.name)


def _ensure_isolated_e2e_db_copy() -> None:
    """Create a per-run DB copy when using a prod snapshot.

    Live E2E runs mutate the sqlite DB (events created/updated, queue statuses,
    scan marks). If we use `db_prod_snapshot.sqlite` directly, the file ceases
    to be a snapshot and subsequent runs become non-deterministic (and can fool
    staleness checks that rely on file mtime).

    Default: enabled. Set E2E_DB_ISOLATE=0 to opt out.
    """
    raw = (os.getenv("E2E_DB_ISOLATE") or "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return

    db_path = (os.environ.get("DB_PATH") or "").strip()
    if not db_path:
        return

    src = Path(db_path)
    if not src.is_absolute():
        # Behave is usually started from repo root; interpret relative DB_PATH
        # as repo-root relative to keep isolation logic predictable.
        src = (project_root / src).resolve()
    if not src.exists() or not src.is_file():
        return

    # Treat any "*snapshot*.sqlite" as a snapshot DB that must be isolated per-run,
    # otherwise live E2E runs mutate the file and become non-repeatable.
    if not (src.suffix == ".sqlite" and "snapshot" in src.name):
        return

    # Use a local filesystem path by default to avoid rare hangs on network/9p mounts
    # during sqlite backup/journal writes. Can be overridden to keep DB copies under
    # artifacts for inspection.
    isolate_dir = (os.getenv("E2E_DB_ISOLATE_DIR") or "").strip()
    if isolate_dir:
        out_dir = Path(isolate_dir)
    else:
        out_dir = Path("/tmp") / "events-bot-new" / "e2e-db"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = out_dir / f"db_e2e_{ts}.sqlite"

    try:
        con_src = sqlite3.connect(str(src))
        con_dst = sqlite3.connect(str(dst))
        try:
            con_src.backup(con_dst)
        finally:
            con_dst.close()
            con_src.close()
    except Exception as exc:
        logger.warning("Failed to isolate E2E DB copy (will use DB_PATH as-is): %s", exc)
        return

    os.environ["E2E_DB_BASE_PATH"] = str(src)
    os.environ["DB_PATH"] = str(dst)
    logger.info("E2E DB isolated copy: %s -> %s", src.name, dst.name)


def _load_dotenv_from_repo_root() -> None:
    """Best-effort .env loader for local dev/E2E runs.

    Behave is often started from an IDE where `.env` is not automatically exported
    into the process environment. We keep this minimal and safe:
    - only sets variables that are currently missing;
    - ignores malformed lines and comments;
    - supports `export KEY=...` and simple quoted values.
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = (k or "").strip()
        if not key:
            continue
        val = (v or "").strip()
        if (len(val) >= 2) and ((val[0] == val[-1]) and val[0] in ("'", '"')):
            val = val[1:-1]
        if not val:
            continue

        # E2E safety: do not override env by default, but some CI/dev shells can export
        # placeholder values (e.g. short Supabase keys) that break media uploads.
        # Supabase anon/service keys are long JWT strings; treat very short values as invalid.
        existing = os.environ.get(key)
        if existing is not None:
            if key in {"SUPABASE_KEY", "SUPABASE_SERVICE_KEY"} and len(existing.strip()) < 80 and len(val) >= 120:
                os.environ[key] = val
                logger.info("E2E dotenv override: %s replaced weak env value from .env", key)
            continue

        os.environ[key] = val


def _resolve_bot_username(*, telethon_client=None) -> str:
    """Resolve bot username for E2E runs.

    Preferred: Telegram Bot API `getMe` (simple and doesn't require Telethon).
    Fallback: resolve by bot user-id via Telethon (works when Bot API is blocked).
    """
    forced = (os.environ.get("E2E_BOT_USERNAME") or "").strip()
    if forced:
        forced = forced.lstrip("@").strip()
        if not forced:
            raise EnvironmentError("E2E_BOT_USERNAME is set but empty after stripping '@'")
        return forced
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN (required for E2E bot username resolution)")
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        # Bot API can be blocked in some environments. Fallback to Telethon using bot user-id.
        if telethon_client is not None:
            try:
                bot_id = int(token.split(":", 1)[0])
                loop = asyncio.get_event_loop()
                entity = loop.run_until_complete(telethon_client.get_entity(bot_id))
                username = (getattr(entity, "username", "") or "").strip()
                if username:
                    return username
            except Exception:
                pass
        raise EnvironmentError(f"Failed to resolve bot username via getMe: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise EnvironmentError(f"Telegram getMe failed: {payload}")
    result = payload.get("result") or {}
    username = (result.get("username") or "").strip()
    if not username:
        raise EnvironmentError("Telegram getMe returned empty username")
    return username


def _maybe_disable_catbox_for_e2e() -> None:
    """Disable Catbox uploads during E2E when the service is unreachable.

    Live E2E should not "hang" on repeated Catbox retries/timeouts. When Catbox
    is unavailable from the current environment, fall back to VK URLs for posters.

    Override:
    - Set CATBOX_FORCE_ENABLED=1/0 explicitly to control behaviour.
    - Set E2E_SKIP_CATBOX_CHECK=1 to skip reachability probing.
    """
    if (os.getenv("CATBOX_FORCE_ENABLED") or "").strip():
        return
    if (os.getenv("E2E_SKIP_CATBOX_CHECK") or "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        req = urllib.request.Request("https://catbox.moe", method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = 200 <= int(getattr(resp, "status", 0) or 0) < 500
    except Exception:
        ok = False
    if not ok:
        os.environ["CATBOX_FORCE_ENABLED"] = "0"
        logger.info("E2E: Catbox unreachable, forcing CATBOX_FORCE_ENABLED=0")


def _ensure_e2e_user_in_db(user_id: int, username: str | None) -> None:
    """Make sure E2E runner user is allowed to use the bot with a prod snapshot DB.

    With a prod DB snapshot there are already users, so /start will not auto-create
    the first user as superadmin. For deterministic E2E we upsert our Telethon user
    into the `user` table and mark it as superadmin (DEV env only).
    """
    db_path = (os.environ.get("DB_PATH") or "").strip()
    if not db_path:
        # Keep legacy behavior for older runs where DB isn't involved.
        return

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        logger.warning("DB open failed for DB_PATH=%s: %s", db_path, exc)
        return

    try:
        row = conn.execute(
            "SELECT user_id, is_superadmin, blocked FROM user WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO user(user_id, username, is_superadmin, is_partner, blocked)
                VALUES (?, ?, 1, 0, 0)
                """,
                (int(user_id), username),
            )
            conn.commit()
            logger.info("Seeded E2E user in DB: user_id=%s", user_id)
            return

        # Keep tests deterministic: ensure not blocked and has superadmin.
        updates = []
        params: list[object] = []
        if int(row["is_superadmin"] or 0) != 1:
            updates.append("is_superadmin = 1")
        if int(row["blocked"] or 0) != 0:
            updates.append("blocked = 0")
        if updates:
            conn.execute(
                f"UPDATE user SET {', '.join(updates)} WHERE user_id = ?",
                (int(user_id),),
            )
            conn.commit()
            logger.info("Updated E2E user flags in DB: user_id=%s updates=%s", user_id, updates)
    except Exception as exc:
        logger.warning("Failed to seed E2E user in DB: %s", exc)
    finally:
        conn.close()


def _cleanup_test_smart_update_data() -> None:
    """Best-effort cleanup for flaky/aborted runs.

    Behave stops executing remaining steps on first failure, so per-scenario
    cleanup steps may not run. Keep E2E idempotent by deleting only the
    synthetic fixtures we create (titles starting with 'TEST SU ').
    """
    db_path = (os.environ.get("DB_PATH") or "").strip()
    if not db_path:
        return
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception:
        return
    try:
        cur = conn.cursor()
        ids = [
            int(r[0])
            for r in cur.execute(
                "SELECT id FROM event WHERE title LIKE 'TEST SU %'"
            ).fetchall()
        ]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        # These tables exist in prod snapshot; ignore if not present.
        for stmt in [
            f"DELETE FROM event_source_fact WHERE event_id IN ({placeholders})",
            f"DELETE FROM event_source WHERE event_id IN ({placeholders})",
            f"DELETE FROM eventposter WHERE event_id IN ({placeholders})",
            f"DELETE FROM event WHERE id IN ({placeholders})",
        ]:
            try:
                cur.execute(stmt, ids)
            except Exception:
                # Table may be missing in older snapshots.
                continue
        conn.commit()
    finally:
        conn.close()

def _tail_text_file(path: Path, max_lines: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _start_local_bot_process() -> tuple[subprocess.Popen, Path]:
    """Start local bot in DEV_MODE polling, so E2E is self-contained.

    Without a running bot process, Telegram will accept /start but nobody will
    reply, which makes E2E look like a timeout/flakiness issue.
    """
    enabled = (os.getenv("E2E_START_LOCAL_BOT") or "1").strip().lower()
    if enabled in {"0", "false", "no"}:
        raise RuntimeError("E2E_START_LOCAL_BOT=0 (local bot autostart disabled)")

    out_dir = project_root / "artifacts" / "test-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_path = out_dir / f"e2e_local_bot_{ts}.log"

    env = os.environ.copy()
    env.setdefault("DEV_MODE", "1")
    # Ensure bot logs are visible immediately for the readiness probe.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Be explicit: DEV mode uses polling; webhook must be empty/ignored.
    env.pop("WEBHOOK_URL", None)

    # Run main.py which exec()s main_part2.py and will start polling in DEV_MODE.
    cmd = [sys.executable, "-u", str(project_root / "main.py")]
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return proc, log_path


def _wait_for_bot_ready(proc: subprocess.Popen, log_path: Path, timeout_sec: int = 90) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    needle = "DEV MODE READY: Bot is now polling for updates"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _tail_text_file(log_path)
            raise RuntimeError(
                "Local bot process exited during startup.\n"
                f"log={log_path}\n"
                f"tail:\n{tail}"
            )
        try:
            if log_path.exists():
                txt = log_path.read_text(encoding="utf-8", errors="replace")
                if needle in txt:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    tail = _tail_text_file(log_path)
    raise RuntimeError(
        "Local bot process did not become ready in time.\n"
        f"log={log_path}\n"
        f"tail:\n{tail}"
    )


def before_all(context):
    """Initialize async event loop and HumanUserClient before all tests."""
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Starting")
    logger.info("=" * 60)
    
    # Create event loop for async operations
    context.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(context.loop)
    # Some environments fail to wake the selector loop for thread-safe callbacks
    # (`call_soon_threadsafe`), which can deadlock libraries that use worker threads
    # (e.g. aiosqlite). Keep a tiny periodic tick so the loop processes callbacks.
    async def _loop_tick():
        import asyncio as _asyncio

        while True:
            await _asyncio.sleep(0.05)

    context._loop_tick_task = context.loop.create_task(_loop_tick())

    # Ensure `.env` variables are available (common in local dev).
    _load_dotenv_from_repo_root()
    # sqlite WAL on some mounts can be flaky under concurrency; DELETE is slower but robust.
    os.environ.setdefault("DB_JOURNAL_MODE", "DELETE")
    _ensure_db_path_env()
    _ensure_isolated_e2e_db_copy()
    _maybe_disable_catbox_for_e2e()

    context.offline = _is_offline_mode()
    if context.offline:
        # Offline mode is meant for DB-only E2E scenarios (no Telegram/Telethon, no network).
        # Speed-up: DB snapshots already contain schema; avoid running full db.init migrations.
        os.environ.setdefault("E2E_SKIP_DB_INIT", "1")
        # Avoid network-only topic classification calls in offline runs.
        os.environ.setdefault("EVENT_TOPICS_LLM", "off")
        context.client = None
        context.bot_username = None
        context.last_message = None
        context.last_response = None
        logger.info("E2E OFFLINE mode enabled (E2E_OFFLINE=1): skipping Telethon/Telegram API setup")
        return

    # Check required environment variables
    # TELEGRAM_API_* are preferred names for E2E, but TG_API_* are used across the repo.
    required = ["TELEGRAM_BOT_TOKEN"]
    missing = [var for var in required if not os.environ.get(var)]
    if not (os.environ.get("TELEGRAM_API_ID") or os.environ.get("TG_API_ID")):
        missing.append("TELEGRAM_API_ID or TG_API_ID")
    if not (os.environ.get("TELEGRAM_API_HASH") or os.environ.get("TG_API_HASH")):
        missing.append("TELEGRAM_API_HASH or TG_API_HASH")
    has_session = bool(os.environ.get("TELEGRAM_SESSION"))
    has_bundle = bool(os.environ.get("TELEGRAM_AUTH_BUNDLE_E2E"))
    if not has_session and not has_bundle:
        missing.append("TELEGRAM_SESSION or TELEGRAM_AUTH_BUNDLE_E2E")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    # Start local bot process by default to avoid silent timeouts when bot isn't running.
    # This can be disabled via E2E_START_LOCAL_BOT=0 for runs against a remotely hosted bot.
    context.bot_proc = None
    context.bot_log_path = None
    try:
        proc, log_path = _start_local_bot_process()
        context.bot_proc = proc
        context.bot_log_path = log_path
        logger.info("Starting local bot process for E2E: log=%s", log_path)
        _wait_for_bot_ready(proc, log_path, timeout_sec=int(os.getenv("E2E_LOCAL_BOT_READY_TIMEOUT_SEC", "90")))
        logger.info("Local bot is ready (polling)")
    except Exception as exc:
        # If local bot can't be started, E2E may still work if bot is hosted elsewhere.
        # Keep running, but log a loud warning for diagnostics.
        logger.warning("Local bot autostart failed: %s", exc)
    
    # Initialize HumanUserClient
    context.client = create_human_client()
    async def _connect_with_retries(max_attempts: int = 10) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                await context.client.connect()
                return
            except Exception as exc:
                last_exc = exc
                try:
                    await context.client.disconnect()
                except Exception:
                    pass
                sleep_s = min(10.0, 0.8 * attempt)
                logger.warning(
                    "Telethon connect failed (attempt %s/%s): %s; retry in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Telethon connect failed: unknown error")

    context.loop.run_until_complete(_connect_with_retries(int(os.getenv("E2E_TELETHON_CONNECT_RETRIES", "10"))))

    # Ensure our Telethon user exists in DB (prod snapshot gating requires it).
    try:
        me = context.loop.run_until_complete(context.client.client.get_me())
        _ensure_e2e_user_in_db(int(me.id), (me.username or me.first_name or "").strip() or None)
    except Exception as exc:
        logger.warning("Failed to ensure E2E user in DB: %s", exc)
    
    # Bot username to test (resolve from token to avoid extra env wiring).
    # Prefer Bot API, but fallback to Telethon when Bot API is blocked/unreachable.
    context.bot_username = _resolve_bot_username(telethon_client=context.client.client)
    
    # Store for last message/response
    context.last_message = None
    context.last_response = None
    
    logger.info(f"Connected! Testing bot: @{context.bot_username}")


def after_all(context):
    """Disconnect client after all tests."""
    if hasattr(context, "client") and context.client:
        context.loop.run_until_complete(context.client.disconnect())
        logger.info("Client disconnected")

    # Stop local bot process if we started one.
    proc = getattr(context, "bot_proc", None)
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
    
    if hasattr(context, "loop") and context.loop:
        tick = getattr(context, "_loop_tick_task", None)
        if tick is not None:
            try:
                tick.cancel()
                context.loop.run_until_complete(asyncio.gather(tick, return_exceptions=True))
            except Exception:
                pass
        context.loop.close()
    
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Finished")
    logger.info("=" * 60)


def before_scenario(context, scenario):
    """Log scenario start."""
    logger.info(f"\n📌 Сценарий: {scenario.name}")
    if getattr(context, "offline", False):
        if "offline" not in getattr(scenario, "effective_tags", []):
            scenario.skip("requires Telegram/Telethon (run without E2E_OFFLINE=1)")
    if "manual" in getattr(scenario, "effective_tags", []):
        if not _is_manual_run_enabled():
            scenario.skip("manual scenario (set E2E_RUN_MANUAL=1 or run with --tags=manual)")
    _cleanup_test_smart_update_data()


def after_scenario(context, scenario):
    """Log scenario result."""
    if scenario.status == "passed":
        logger.info(f"✅ Сценарий PASSED: {scenario.name}")
    elif scenario.status == "skipped":
        logger.warning(f"⏭️ Сценарий SKIPPED: {scenario.name} ({scenario.skip_reason})")
    else:
        logger.error(f"❌ Сценарий FAILED: {scenario.name}")
