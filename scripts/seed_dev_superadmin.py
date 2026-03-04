#!/usr/bin/env python3
"""
Seed DEV superadmin for local/live E2E.

Use-cases:
- Fresh prod snapshot DB does not include your test Telegram user -> /vk replies "Access denied".
- You want deterministic rights in local DB copy without touching prod.

How it works:
1) Logs in via Telethon using TELEGRAM_AUTH_BUNDLE_E2E (preferred) or TELEGRAM_SESSION.
2) Reads current Telegram user_id/username.
3) Upserts row into sqlite `user` table (DB_PATH) and sets is_superadmin=1, blocked=0.

Safety:
- Intended for DEV/local usage only (does not touch prod DB).
- Does not print secrets (session strings), only prints a short confirmation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sqlite3
from dataclasses import dataclass

from telethon import TelegramClient
from telethon.sessions import StringSession


@dataclass(frozen=True, slots=True)
class TelethonConfig:
    api_id: int
    api_hash: str
    session_string: str
    device_model: str | None = None
    system_version: str | None = None
    app_version: str | None = None
    lang_code: str | None = None
    system_lang_code: str | None = None


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _load_telethon_config() -> TelethonConfig:
    api_id_raw = _env("TELEGRAM_API_ID") or _env("TG_API_ID")
    api_hash = _env("TELEGRAM_API_HASH") or _env("TG_API_HASH")
    if not api_id_raw or not api_hash:
        raise SystemExit("Missing TELEGRAM_API_ID/TELEGRAM_API_HASH (or TG_API_ID/TG_API_HASH)")

    bundle_b64 = _env("TELEGRAM_AUTH_BUNDLE_E2E")
    session_string = _env("TELEGRAM_SESSION")
    if bundle_b64:
        raw = base64.urlsafe_b64decode(bundle_b64.encode("ascii")).decode("utf-8")
        bundle = json.loads(raw)
        session_string = (bundle.get("session") or "").strip()
        if not session_string:
            raise SystemExit("Invalid TELEGRAM_AUTH_BUNDLE_E2E: missing session")
        return TelethonConfig(
            api_id=int(api_id_raw),
            api_hash=api_hash,
            session_string=session_string,
            device_model=(bundle.get("device_model") or None),
            system_version=(bundle.get("system_version") or None),
            app_version=(bundle.get("app_version") or None),
            lang_code=(bundle.get("lang_code") or None),
            system_lang_code=(bundle.get("system_lang_code") or None),
        )

    if not session_string:
        raise SystemExit("Missing TELEGRAM_AUTH_BUNDLE_E2E or TELEGRAM_SESSION")
    return TelethonConfig(api_id=int(api_id_raw), api_hash=api_hash, session_string=session_string)


async def _resolve_me(cfg: TelethonConfig) -> tuple[int, str | None]:
    kwargs: dict[str, object] = {}
    for key in ("device_model", "system_version", "app_version", "lang_code", "system_lang_code"):
        val = getattr(cfg, key)
        if val:
            kwargs[key] = val
    async with TelegramClient(StringSession(cfg.session_string), cfg.api_id, cfg.api_hash, **kwargs) as client:
        me = await client.get_me()
        uid = int(me.id)
        username = (getattr(me, "username", None) or getattr(me, "first_name", None) or "").strip() or None
        return uid, username


def _upsert_superadmin(db_path: str, user_id: int, username: str | None) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        row = conn.execute(
            "SELECT user_id FROM user WHERE user_id = ?",
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
        else:
            conn.execute(
                "UPDATE user SET is_superadmin = 1, blocked = 0 WHERE user_id = ?",
                (int(user_id),),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_env("DB_PATH") or "db_prod_snapshot.sqlite")
    args = ap.parse_args()

    db_path = (args.db or "").strip()
    if not db_path:
        raise SystemExit("DB_PATH is not set (use --db or export DB_PATH)")

    cfg = _load_telethon_config()
    user_id, username = asyncio.run(_resolve_me(cfg))
    _upsert_superadmin(db_path, user_id=user_id, username=username)
    print("OK: seeded DEV superadmin for Telethon user (id hidden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

