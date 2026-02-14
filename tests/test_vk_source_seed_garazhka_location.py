from __future__ import annotations

import aiosqlite
import pytest

from db import Database


@pytest.mark.asyncio
async def test_db_init_seeds_garazhka_vk_source_location(tmp_path) -> None:
    path = tmp_path / "db.sqlite"
    async with aiosqlite.connect(str(path)) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vk_source(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                screen_name TEXT,
                name TEXT,
                location TEXT,
                default_time TEXT,
                default_ticket_link TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_vk_source_group ON vk_source(group_id)"
        )
        await conn.execute(
            """
            INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link)
            VALUES(?,?,?,?,?,?)
            """,
            (226847232, "garazhka_kld", "Garazhka Kaliningrad", None, None, None),
        )
        await conn.commit()

    db = Database(str(path))
    await db.init()
    async with aiosqlite.connect(str(path)) as conn:
        cur = await conn.execute(
            "SELECT location FROM vk_source WHERE group_id=?",
            (226847232,),
        )
        (location,) = await cur.fetchone()
    assert location == "Понарт, Судостроительная 6/2, Калининград"

    # Must not override operator-set locations after the initial seed.
    async with aiosqlite.connect(str(path)) as conn:
        await conn.execute(
            "UPDATE vk_source SET location=? WHERE group_id=?",
            ("Другая локация", 226847232),
        )
        await conn.commit()

    await db.init()
    async with aiosqlite.connect(str(path)) as conn:
        cur = await conn.execute(
            "SELECT location FROM vk_source WHERE group_id=?",
            (226847232,),
        )
        (location,) = await cur.fetchone()
    assert location == "Другая локация"
