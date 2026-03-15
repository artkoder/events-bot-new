from __future__ import annotations

import pytest

from admin_chat import resolve_superadmin_chat_id
from db import Database


@pytest.mark.asyncio
async def test_resolve_superadmin_chat_id_prefers_database_over_env(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            'INSERT INTO "user"(user_id, username, is_superadmin, blocked) VALUES(?, ?, 1, 0)',
            (185169715, "max",),
        )
        await conn.commit()

    monkeypatch.setenv("ADMIN_CHAT_ID", "999999")

    assert await resolve_superadmin_chat_id(db) == 185169715


@pytest.mark.asyncio
async def test_resolve_superadmin_chat_id_falls_back_to_env_when_db_empty(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("ADMIN_CHAT_ID", "777777")

    assert await resolve_superadmin_chat_id(db) == 777777
