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

def _resolve_bot_username() -> str:
    """Resolve bot username from TELEGRAM_BOT_TOKEN via Telegram Bot API."""
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN (required for E2E bot username resolution)")
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise EnvironmentError(f"Failed to resolve bot username via getMe: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        raise EnvironmentError(f"Telegram getMe failed: {payload}")
    result = payload.get("result") or {}
    username = (result.get("username") or "").strip()
    if not username:
        raise EnvironmentError("Telegram getMe returned empty username")
    return username


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


def before_all(context):
    """Initialize async event loop and HumanUserClient before all tests."""
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Starting")
    logger.info("=" * 60)
    
    # Create event loop for async operations
    context.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(context.loop)
    
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
    
    # Initialize HumanUserClient
    context.client = create_human_client()
    context.loop.run_until_complete(context.client.connect())

    # Ensure our Telethon user exists in DB (prod snapshot gating requires it).
    try:
        me = context.loop.run_until_complete(context.client.client.get_me())
        _ensure_e2e_user_in_db(int(me.id), (me.username or me.first_name or "").strip() or None)
    except Exception as exc:
        logger.warning("Failed to ensure E2E user in DB: %s", exc)
    
    # Bot username to test (resolve from token to avoid extra env wiring)
    context.bot_username = _resolve_bot_username()
    
    # Store for last message/response
    context.last_message = None
    context.last_response = None
    
    logger.info(f"Connected! Testing bot: @{context.bot_username}")


def after_all(context):
    """Disconnect client after all tests."""
    if hasattr(context, "client") and context.client:
        context.loop.run_until_complete(context.client.disconnect())
        logger.info("Client disconnected")
    
    if hasattr(context, "loop") and context.loop:
        context.loop.close()
    
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Finished")
    logger.info("=" * 60)


def before_scenario(context, scenario):
    """Log scenario start."""
    logger.info(f"\n📌 Сценарий: {scenario.name}")
    _cleanup_test_smart_update_data()


def after_scenario(context, scenario):
    """Log scenario result."""
    if scenario.status == "passed":
        logger.info(f"✅ Сценарий PASSED: {scenario.name}")
    else:
        logger.error(f"❌ Сценарий FAILED: {scenario.name}")
